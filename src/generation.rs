use crate::model::{build_alibi_tensor, Config, Past};
use crate::non_empty_past;
use log::info;
use serde::{Deserialize, Serialize};
use tch::{kind, IndexOp, Tensor};

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct Sampling {
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone)]
pub struct BeamSearch {
    pub num_beams: usize,
}

pub fn default_temperature() -> f64 {
    1.0
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone)]
pub struct Greedy {}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub enum GenerationMode {
    BeamSearch(BeamSearch),
    Sampling(Sampling),
    Greedy,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
#[serde(try_from = "IntermediateParameters")]
pub struct Parameters {
    pub generation_mode: GenerationMode,
    pub max_new_tokens: usize,
}

impl TryFrom<IntermediateParameters> for Parameters {
    // TODO: impl proper error type instead of `String`
    type Error = String;

    fn try_from(data: IntermediateParameters) -> Result<Self, Self::Error> {
        let generation_mode = match (data.do_sample, data.num_beams) {
            (Some(do_sample), _) if do_sample => GenerationMode::Sampling(Sampling {
                temperature: data.temperature.unwrap_or(1.0),
                top_k: data.top_k,
                top_p: data.top_p,
            }),
            (_, Some(num_beams)) => GenerationMode::BeamSearch(BeamSearch { num_beams }),
            _ => {
                if let Some(temperature) = data.temperature {
                    GenerationMode::Sampling(Sampling {
                        temperature,
                        top_k: data.top_k,
                        top_p: data.top_p,
                    })
                } else if let Some(_top_k) = data.top_k {
                    GenerationMode::Sampling(Sampling {
                        temperature: data.temperature.unwrap_or(1.0),
                        top_k: data.top_k,
                        top_p: data.top_p,
                    })
                } else if let Some(_top_p) = data.top_p {
                    GenerationMode::Sampling(Sampling {
                        temperature: data.temperature.unwrap_or(1.0),
                        top_k: data.top_k,
                        top_p: data.top_p,
                    })
                } else {
                    GenerationMode::Greedy
                }
            }
        };
        Ok(Parameters {
            generation_mode,
            max_new_tokens: data.max_new_tokens.unwrap_or(20),
        })
    }
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            generation_mode: GenerationMode::Greedy,
            max_new_tokens: 20,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct IntermediateParameters {
    do_sample: Option<bool>,
    num_beams: Option<usize>,
    top_p: Option<f64>,
    top_k: Option<usize>,
    temperature: Option<f64>,
    max_new_tokens: Option<usize>,
}

pub fn filter_top_p(scored_logits: &Tensor, top_p: f64) -> Tensor {
    let filter_value = f64::NEG_INFINITY;
    let descending = true;
    let (sorted_logits, sorted_indices) = scored_logits.sort(-1, descending);
    // cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    let cumulative_probs = sorted_logits
        .softmax(-1, kind::Kind::Float)
        .cumsum(-1, kind::Kind::Float);

    // # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    // sorted_indices_to_remove = cumulative_probs > self.top_p
    let mut sorted_indices_to_remove = cumulative_probs.gt(top_p);
    // if self.min_tokens_to_keep > 1:
    //     # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
    //     sorted_indices_to_remove[..., : self.min_tokens_to_keep - 1] = 0
    // # Shift the indices to the right to keep also the first token above the threshold
    // sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    // sorted_indices_to_remove[..., 0] = 0

    // sorted_indices_to_remove = sorted_indices_to_remove
    //     .i((0..B, 1..S))
    //     .fill_tensor(&sorted_indices_to_remove.i((0..B, 0..S - 1)).view((-1)));
    let _ = sorted_indices_to_remove.i((.., -1)).f_fill_(0).unwrap();
    sorted_indices_to_remove = sorted_indices_to_remove.roll(&[1], &[1]);

    // # scatter sorted tensors to original indexing
    // indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    let indices_to_remove = sorted_indices_to_remove
        .f_scatter(1, &sorted_indices, &sorted_indices_to_remove)
        .unwrap();
    scored_logits
        .f_masked_fill(&indices_to_remove, filter_value)
        .unwrap()
    // scores = scores.masked_fill(indices_to_remove, self.filter_value)
    // return scores
}

pub fn next_ids(params: &Parameters, logits: &Tensor) -> Tensor {
    // TODO handle batching
    match &params.generation_mode {
        GenerationMode::Greedy => {
            let seq_length = logits.size()[1];
            logits
                .i((0..1, seq_length - 1..seq_length))
                .argmax(-1, false)
        }
        GenerationMode::Sampling(params) => {
            let seq_length = logits.size()[1];
            let filter_value = f64::NEG_INFINITY;
            let last_logits = logits.i((0, seq_length - 1..seq_length));

            let mut scored_logits = last_logits / params.temperature;

            if let Some(top_k) = params.top_k {
                // indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
                let largest = true;
                let sorted = true;
                let top_ks = scored_logits.topk(top_k as i64, -1, largest, sorted).0;
                let top_k = top_ks.i((.., -1));

                let indices_to_remove = scored_logits.le_tensor(&top_k);
                scored_logits = scored_logits.masked_fill(&indices_to_remove, filter_value);
            }

            if let Some(top_p) = params.top_p {
                // sorted_logits, sorted_indices = torch.sort(scores, descending=True)

                scored_logits = filter_top_p(&scored_logits, top_p);
            }

            let probs = scored_logits.f_softmax(-1, kind::Kind::Float).unwrap();
            probs.f_multinomial(1, false).unwrap()
        }
        _ => todo!(),
        // Parameters::BeamSearch(params) => {
        //     let num_beams = params.num_beams as i64;
        //     if self.new_generated_tokens == 0 {
        //         // We're in the first step it's rather easy.
        //         let S = logits.size()[1];
        //         let last_logits = logits.i((0, S - 1..S)).to_device(self.input_ids.device());
        //         // Actually cast to logits so we can save the scores
        //         let last_logits = last_logits.f_log_softmax(-1, kind::Kind::Float).unwrap();
        //         let largest = true;
        //         let sorted = true;
        //         let top_ks = last_logits.topk(num_beams, -1, largest, sorted);
        //         let values = top_ks.0;
        //         let indices = top_ks.1;

        //         // repeat input_ids to fit the new tokens
        //         let input_ids = self.input_ids.repeat(&[num_beams, 1]);
        //         let new_ids = indices.to_device(self.input_ids.device());

        //         // new_ids is now of shape [1, num_beams]
        //         // input_ids is of shape [num_beams, seq_length]
        //         // So transposing to we can concatenate the indices within input_ids
        //         let new_ids = new_ids.transpose(1, 0);
        //         self.input_ids = Tensor::f_cat(&[input_ids.copy(), new_ids.copy()], 1).unwrap();
        //         // Save the current scores in logits form.
        //         self.beam_scores = Some(values);
        //     } else {
        //         // Now the tricky part.
        //         let size = logits.size();
        //         let last_logits = logits
        //             .i((0..size[0], size[1] - 1..size[1]))
        //             .to_device(self.input_ids.device());
        //         // Actually cast to logits so we can save the scores
        //         let last_logits = last_logits.f_log_softmax(-1, kind::Kind::Float).unwrap();
        //         panic!("We haven't handled that part !");
        //     }
        // }
    }
}

pub fn padding(config: &Config, items: Vec<(Tensor, Past)>, world_size: usize) -> (Tensor, Tensor, Tensor, Past) {
    let (max_length_input_ids, max_length_past) = items
        .iter()
        .map(|(ids, past)| (ids.size()[1], past[0].seq_length()))
        .max()
        .unwrap();

    match (max_length_input_ids, max_length_past){
        (1, _) => (), // Past is used only 1 new input_id
        (_, 0) => (), // No past is used
        (n, m) => panic!("Padding can handle this situation but it's highly irregular so please fix your call input_ids length {n}, past_length {m}"),
    }

    let max_length = max_length_input_ids + max_length_past;

    let batch_size: i64 = items.iter().map(|(ids, _)| ids.size()[0]).sum::<i64>();
    let device = items[0].0.device();
    let kind2 = (kind::Kind::Int64, device);
    let all_input_ids =
        Tensor::zeros(&[batch_size, max_length_input_ids], kind2) + config.padding_idx;

    let mut all_past_key_values = non_empty_past(config, batch_size, max_length_past, 0.0, 0.0, world_size);

    let attention_mask = Tensor::zeros(&[batch_size, max_length], kind2);

    let mut total_ids = 0;

    let mut current_batch = 0;
    for (input_ids, past_key_values) in items {
        let seq_length = input_ids.size()[1];
        let past_seq_length = past_key_values[0].seq_length();
        let mini_batch_size = input_ids.size()[0];
        total_ids += mini_batch_size as usize * (seq_length + past_seq_length) as usize;

        // all_input_ids[i:i+mini_batch_size, max_length - seq_length:seq_length] =
        // input_ids
        all_input_ids
            .i((
                current_batch..current_batch + mini_batch_size,
                max_length_input_ids - seq_length..,
            ))
            .f_copy_(&input_ids)
            .unwrap();

        _ = attention_mask
            .i((
                current_batch..current_batch + mini_batch_size,
                max_length - seq_length..,
            ))
            .f_fill_(1)
            .unwrap();

        if max_length_past > 0 {
            for (i, layer_past) in all_past_key_values.iter_mut().enumerate() {
                let device = past_key_values[i].key.device();
                if layer_past.key.device() != device {
                    layer_past.key = layer_past.key.to_device(device);
                    layer_past.value = layer_past.value.to_device(device);
                }
                let start_batch_size_times_num_heads = current_batch * config.n_head;
                let end_batch_size_times_num_heads =
                    start_batch_size_times_num_heads + mini_batch_size * config.n_head;
                layer_past
                    .key
                    .i((
                        start_batch_size_times_num_heads..end_batch_size_times_num_heads,
                        ..,
                        max_length_past - past_seq_length..,
                    ))
                    .f_copy_(&past_key_values[i].key)
                    .unwrap();

                layer_past
                    .value
                    .i((
                        start_batch_size_times_num_heads..end_batch_size_times_num_heads,
                        max_length_past - past_seq_length..,
                    ))
                    .f_copy_(&past_key_values[i].value)
                    .unwrap();
            }
            _ = attention_mask
                .i((
                    current_batch..current_batch + mini_batch_size,
                    max_length_past - past_seq_length..max_length_past,
                ))
                .f_fill_(1)
                .unwrap();
        }

        current_batch += mini_batch_size;
    }

    let alibi = build_alibi_tensor(&attention_mask, config.n_head, config.kind, device);

    let total = std::cmp::max(1, batch_size as usize * max_length as usize);
    info!(
        "Running on batch of size, seq_length[{:?}, {:?}] - Fillrate {:?}%",
        batch_size,
        max_length,
        (total_ids * 100) / total
    );
    (all_input_ids, attention_mask, alibi, all_past_key_values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::empty_past;
    use tch::Device;

    #[test]
    fn test_sampling_top_p() {
        // dist = torch.log(
        //     torch.tensor([[0.3, 0.1, 0.1, 0.5], [0.15, 0.3, 0.3, 0.25]], device=torch_device, dtype=torch.float)
        // )
        let top_p = 0.7;
        let dist = Tensor::of_slice(&[0.3, 0.1, 0.1, 0.5, 0.15, 0.3, 0.3, 0.25]).view((2, 4));

        // top_p_warp = TopPLogitsWarper(0.7)
        // filtered_dist = torch.exp(top_p_warp(input_ids, dist))
        let filtered_dist = filter_top_p(&dist.log(), top_p).exp();

        // # dist should be filtered to keep min num values so that sum is >= 0.7
        // # exp (-inf) => 0
        // EXPECTED_FILTERED_DIST = torch.tensor(
        //     [[0.3, 0.0, 0.0, 0.5], [0.0, 0.3, 0.3, 0.25]], device=torch_device, dtype=torch.float
        // )
        let expected = Tensor::of_slice(&[0.3, 0.0, 0.0, 0.5, 0.0, 0.3, 0.3, 0.25]).view((2, 4));

        // self.assertTrue(torch.allclose(filtered_dist, EXPECTED_FILTERED_DIST, atol=1e-3))
        let rtol = 1e-05;
        let atol = 1e-03;
        let equal_nan = false;
        assert!(filtered_dist.allclose(&expected, rtol, atol, equal_nan));
    }
    #[test]
    fn test_unpacking_params() {
        let parameters: Parameters = serde_json::from_str(r#"{"do_sample": true}"#).unwrap();
        assert_eq!(
            parameters,
            Parameters {
                generation_mode: GenerationMode::Sampling(Sampling {
                    top_p: None,
                    top_k: None,
                    temperature: 1.0,
                }),
                max_new_tokens: 20
            }
        );
        let parameters: Parameters = serde_json::from_str(r#"{}"#).unwrap();
        assert_eq!(
            parameters,
            Parameters {
                generation_mode: GenerationMode::Greedy,
                max_new_tokens: 20
            }
        );
        let parameters: Parameters = serde_json::from_str(r#"{"do_sample": false}"#).unwrap();
        assert_eq!(
            parameters,
            Parameters {
                generation_mode: GenerationMode::Greedy,
                max_new_tokens: 20
            }
        );
        let parameters: Parameters = serde_json::from_str(r#"{"top_k": 20}"#).unwrap();
        assert_eq!(
            parameters,
            Parameters {
                generation_mode: GenerationMode::Sampling(Sampling {
                    top_p: None,
                    top_k: Some(20),
                    temperature: 1.0,
                }),
                max_new_tokens: 20
            }
        );
    }
    #[test]
    fn test_padding() {
        let config = Config::new350m();
        let device = Device::Cuda(0);
        let input_ids = Tensor::of_slice(&[3, 4, 5])
            .view((1, 3))
            .to_kind(kind::Kind::Int64)
            .to_device(device);
        let input_ids2 = Tensor::of_slice(&[8, 1, 3, 4, 5, 6])
            .view((1, 6))
            .to_kind(kind::Kind::Int64)
            .to_device(device);
        let past = empty_past(&config, 1, 1);
        let past2 = empty_past(&config, 1, 1);

        let items = vec![(input_ids, past), (input_ids2, past2)];

        let (all_input_ids, _, _, _) = padding(&config, items, 1);

        assert_eq!(all_input_ids.size(), vec![2, 6]);
        assert_eq!(
            Vec::<i64>::from(all_input_ids),
            vec![3, 3, 3, 3, 4, 5, 8, 1, 3, 4, 5, 6]
        );
    }

    #[test]
    fn test_padding_past() {
        let config = Config::new350m();
        let device = Device::Cuda(0);
        let input_ids = Tensor::of_slice(&[3])
            .view((1, 1))
            .to_kind(kind::Kind::Int64)
            .to_device(device);
        let input_ids2 = Tensor::of_slice(&[8])
            .view((1, 1))
            .to_kind(kind::Kind::Int64)
            .to_device(device);

        // Batch_size=1, seq_length=4
        let past = non_empty_past(&config, 1, 4, 3.0, 4.0, 1);
        let past2 = non_empty_past(&config, 1, 6, 5.0, 6.0, 1);

        let items = vec![(input_ids, past), (input_ids2, past2)];

        let (all_input_ids, attention_mask, alibi, past_key_values) = padding(&config, items,1);

        assert_eq!(all_input_ids.size(), vec![2, 1]);
        assert_eq!(attention_mask.size(), vec![2, 7]);
        assert_eq!(alibi.size(), vec![16 * 2, 1, 7]);
        assert_eq!(past_key_values.len(), 24);
        for i in 0..24 {
            assert_eq!(past_key_values[i].key.size(), vec![32, 64, 6], "Layer {i}");
            assert_eq!(
                past_key_values[i].value.size(),
                vec![32, 6, 64],
                "Layer {i}"
            );
        }

        assert_eq!(Vec::<i64>::from(all_input_ids), vec![3, 8]);
        assert_eq!(
            Vec::<i64>::from(attention_mask.i(0)),
            vec![0, 0, 1, 1, 1, 1, 1]
        );
        assert_eq!(
            Vec::<i64>::from(attention_mask.i(1)),
            vec![1, 1, 1, 1, 1, 1, 1]
        );
        assert_eq!(
            Vec::<f64>::from(past_key_values[0].key.i((0..1, ..1, ..))),
            vec![0.0, 0.0, 3.0, 3.0, 3.0, 3.0]
        );
        assert_eq!(
            Vec::<f64>::from(past_key_values[0].value.i((0..1, .., ..1))),
            vec![0.0, 0.0, 4.0, 4.0, 4.0, 4.0]
        );
        assert_eq!(
            Vec::<f64>::from(past_key_values[0].key.i((16..17, ..1, ..))),
            vec![5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        );
        assert_eq!(
            Vec::<f64>::from(past_key_values[0].value.i((16..17, .., ..1))),
            vec![6.0, 6.0, 6.0, 6.0, 6.0, 6.0]
        );
    }
}
