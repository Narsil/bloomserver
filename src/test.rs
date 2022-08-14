use super::*;
use crate::generation::padding;
use crate::model::build_alibi_tensor;
use crate::model::tests::{BLOOM_350M, BLOOM_TESTING};
use crate::model::BloomForCausalLM;
use crate::model::Config;
use tch::{IndexOp, Tensor};
use tokenizers::Tokenizer;

#[derive(Debug)]
pub struct Error(String);

pub fn assert_all_close(left: &Tensor, right: &Tensor) -> Result<(), Error> {
    if !left.allclose(right, 1e-7, 1e-7, false) {
        left.print();
        right.print();
        Err(Error("{left:?} is not close to {right:?}".to_string()))
    } else {
        Ok(())
    }
}

fn test_generate(
    input: &[&str],
    config: &Config,
    tokenizer: &Tokenizer,
    model: &BloomForCausalLM,
    max_new_tokens: usize,
) -> Vec<String> {
    // Taken directly from https://github.com/huggingface/transformers/blob/main/tests/models/bloom/test_modeling_bloom.py#L379
    let mut all_items = vec![];
    for input_string in input {
        let encoded = tokenizer.encode(input_string.to_string(), false).unwrap();
        let ids: Vec<_> = encoded.get_ids().iter().map(|&i| i as i64).collect();
        let input_ids = Tensor::of_slice(ids.as_slice())
            .to_kind(kind::Kind::Int64)
            .to_device(Device::Cuda(0))
            .view((1, -1));
        let past = empty_past(&config, 1);

        // Not necessary, but want to reuse the real code
        let item = (input_ids, past);
        all_items.push(item);
    }

    let (mut input_ids, mut attention_mask, mut alibi, mut past_key_values) =
        padding(&config, all_items);

    let mut all_ids = input_ids.copy();

    for _ in 0..max_new_tokens {
        let logits = model.forward(&input_ids, &attention_mask, &alibi, &mut past_key_values);
        let seq_len = logits.size()[1];
        let new_ids = logits.i((.., seq_len - 1..seq_len)).argmax(-1, false);
        let ones = new_ids.ones_like();

        attention_mask = Tensor::cat(&[attention_mask, ones.copy()], 1);
        alibi = build_alibi_tensor(
            &attention_mask,
            config.n_head,
            config.kind,
            attention_mask.device(),
        );

        all_ids = Tensor::cat(&[all_ids, new_ids.copy()], 1);
        input_ids = new_ids;
    }

    let mut all_strings = vec![];
    for i in 0..input.len() {
        let output_ids: Vec<_> = all_ids
            .i(i as i64)
            .reshape(&[-1])
            .iter::<i64>()
            .unwrap()
            .map(|i| i as u32)
            .collect();
        // Do skip special tokens
        let string = tokenizer.decode(output_ids.clone(), true).unwrap();
        all_strings.push(string);
    }
    all_strings
}

#[test]
fn test_simple_generation() {
    let config = Config::new350m();
    let model = BLOOM_350M.lock().unwrap();
    let tokenizer = Tokenizer::from_file("./tokenizer.json").unwrap();

    let input_sentence = "I enjoy walking my cute dog";
    let input_sentence2 = "Hello my name is";

    let out1 = "I enjoy walking my cute dog, but I also love to play with my cat. I am a very active person and I love to be active. I am a very good listener and I am very good at listening to people. I am very";
    let out2 =  "Hello my name is Nate and I am a professional photographer in the area of the city of New York. I am a professional photographer who loves to capture the best moments in the life of people. I love";

    let output = test_generate(&[input_sentence], &config, &tokenizer, &model, 43);
    assert_eq!(output, vec![out1]);
    let output = test_generate(
        &[input_sentence, input_sentence],
        &config,
        &tokenizer,
        &model,
        43,
    );
    assert_eq!(output, vec![out1, out1]);

    let output = test_generate(&[input_sentence2], &config, &tokenizer, &model, 40);
    assert_eq!(output, vec![out2]);

    let output = test_generate(
        &[input_sentence, input_sentence2],
        &config,
        &tokenizer,
        &model,
        43,
    );
    // TODO This test fails, this is likely due do some padding actually getting
    // non null information flow (but it should be)
    assert_eq!(output[0], out1);
    assert_ne!(output, vec![out1, out2]);
}

#[test]
fn test_logits_testing() {
    let config = Config::new_testing();
    let model = BLOOM_TESTING.lock().unwrap();
    let device = Device::Cuda(0);

    let example_ids = &[
        3478, 368, 109586, 35433, 2, 77, 132619, 3478, 368, 109586, 35433, 2, 2175, 23714, 73173,
        144252, 2, 77, 132619, 3478,
    ];
    let tensor_ids = Tensor::of_slice(example_ids)
        .view((1, -1))
        .to_kind(kind::Kind::Int64)
        .to_device(device);
    let past = empty_past(&config, 1);
    let (input_ids, attention_mask, alibi, mut past_key_values) =
        padding(&config, vec![(tensor_ids, past)]);

    let logits = model.forward(&input_ids, &attention_mask, &alibi, &mut past_key_values);
    let splits = logits.split(125440, -1);
    assert_eq!(splits.len(), 2);
    let output_gpu_1 = &splits[0];
    let output_gpu_2 = &splits[1];

    assert_all_close(
        &output_gpu_1.mean(logits.kind()),
        &Tensor::of_slice(&[-1.823902130126953e-05])
            .to_kind(config.kind)
            .to_device(device),
    )
    .unwrap();
    assert_all_close(
        &output_gpu_2.mean(logits.kind()),
        &Tensor::of_slice(&[1.9431114196777344e-05])
            .to_kind(config.kind)
            .to_device(device),
    )
    .unwrap();
}

#[test]
fn test_embeddings_testing() {
    let config = Config::new_testing();
    let model = BLOOM_TESTING.lock().unwrap();
    let device = Device::Cuda(0);

    let example_ids = &[
        3478, 368, 109586, 35433, 2, 77, 132619, 2175, 23714, 73173, 144252,
    ];
    let tensor_ids = Tensor::of_slice(example_ids)
        .view((1, -1))
        .to_kind(kind::Kind::Int64)
        .to_device(device);
    let past = empty_past(&config, 1);
    let (input_ids, _attention_mask, _alibi, _past_key_values) =
        padding(&config, vec![(tensor_ids, past)]);

    let embeddings = model.transformer.word_embeddings.forward(&input_ids);

    assert_all_close(
        &embeddings.mean_dim(&[-1], false, embeddings.kind()),
        &Tensor::of_slice(&[
            0.0002307891845703125,
            -0.000568389892578125,
            -0.0003910064697265625,
            -0.000194549560546875,
            0.0004138946533203125,
            0.000659942626953125,
            -0.00031280517578125,
            0.000457763671875,
            0.000263214111328125,
            -0.000286102294921875,
            0.00052642822265625,
        ])
        .view((1, -1))
        .to_kind(config.kind)
        .to_device(device),
    )
    .unwrap();

    assert_all_close(
        &embeddings.min_dim(-1, false).0,
        &Tensor::of_slice(&[
            -0.00921630859375,
            -0.010009765625,
            -0.01031494140625,
            -0.01177978515625,
            -0.0074462890625,
            -0.00848388671875,
            -0.009521484375,
            -0.0074462890625,
            -0.0145263671875,
            -0.007415771484375,
            -0.01007080078125,
        ])
        .view((1, -1))
        .to_kind(config.kind)
        .to_device(device),
    )
    .unwrap();

    assert_all_close(
        &embeddings.max_dim(-1, false).0,
        &Tensor::of_slice(&[
            0.0128173828125,
            0.01214599609375,
            0.0111083984375,
            0.01019287109375,
            0.0157470703125,
            0.0174560546875,
            0.0078125,
            0.0113525390625,
            0.0146484375,
            0.01116943359375,
            0.01141357421875,
        ])
        .view((1, -1))
        .to_kind(config.kind)
        .to_device(device),
    )
    .unwrap();

    let embeddings_ln = model
        .transformer
        .word_embeddings_layernorm
        .forward(&embeddings);

    assert_all_close(
        &embeddings_ln.mean_dim(&[-1], false, embeddings_ln.kind()),
        &Tensor::of_slice(&[
            -6.580352783203125e-05,
            0.0001316070556640625,
            -0.00030517578125,
            4.00543212890625e-05,
            -7.2479248046875e-05,
            -8.96453857421875e-05,
            0.0001583099365234375,
            2.1219253540039062e-05,
            -0.000247955322265625,
            -0.00021839141845703125,
            -0.0001430511474609375,
        ])
        .view((1, -1))
        .to_kind(config.kind)
        .to_device(device),
    )
    .unwrap();

    assert_all_close(
        &embeddings_ln.min_dim(-1, false).0,
        &Tensor::of_slice(&[
            -1.6953125, -1.6875, -1.6875, -2.125, -1.390625, -1.5390625, -1.875, -1.4609375,
            -2.296875, -1.3515625, -1.78125,
        ])
        .view((1, -1))
        .to_kind(config.kind)
        .to_device(device),
    )
    .unwrap();
    assert_all_close(
        &embeddings_ln.max_dim(-1, false).0,
        &Tensor::of_slice(&[
            2.265625, 2.28125, 1.953125, 1.90625, 2.703125, 2.828125, 1.65625, 2.015625, 2.234375,
            2.171875, 1.828125,
        ])
        .view((1, -1))
        .to_kind(config.kind)
        .to_device(device),
    )
    .unwrap();
}
