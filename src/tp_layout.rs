use crate::layout::padding_with_ack;
use crate::layout::RChan1;
use crate::model::Config;
use crossbeam_channel::{Receiver, Select, Sender};
use nccl_rs::ThreadGroup;
use tch::IndexOp;

pub fn thread(group: ThreadGroup, config: Config, channels: Option<(RChan1, RChan1)>) {
    // TODO load the weights
    if let Some((rx, prio_rx)) = channels {
        loop {
            let mut sel = Select::new();
            let oper1 = sel.recv(&rx);
            let oper2 = sel.recv(&prio_rx);
            let oper = sel.select();
            let (mut all_items, no_past) = match oper.index() {
                i if i == oper1 => (vec![oper.recv(&rx).unwrap()], true),
                i if i == oper2 => (vec![oper.recv(&prio_rx).unwrap()], false),
                _ => unreachable!(),
            };

            if no_past {
                let max_batch_size = 4;
                while let Ok(item) = prio_rx.recv() {
                    all_items.push(item);
                    if all_items.len() >= max_batch_size {
                        break;
                    }
                }
            }
            let ((input_ids, attention_mask, alibi, mut past_key_values), acks) =
                padding_with_ack(&config, all_items);
            // TODO implement forward
            // let mut current_batch = 0 as i64;
            // for (mini_batch_size, rq) in rqs {
            //     // XXX actually clean the padded values of past so that subsequent
            //     // calls can get a chance to have a better padding (+ correct attention mask).
            //     let seq_length = seq_lengths[current_batch as usize];
            //     let total_seq_length = causal_mask.size()[2];
            //     let start_batch_size_times_num_heads = current_batch * config.n_head;
            //     let end_batch_size_times_num_heads =
            //         start_batch_size_times_num_heads + mini_batch_size * config.n_head;
            //     let past: Vec<_> = past_key_values
            //         .iter()
            //         .map(|layer_past| PastLayer {
            //             key: layer_past.key.i((
            //                 start_batch_size_times_num_heads..end_batch_size_times_num_heads,
            //                 ..,
            //                 total_seq_length - seq_length..,
            //             )),
            //             value: layer_past.value.i((
            //                 start_batch_size_times_num_heads..end_batch_size_times_num_heads,
            //                 total_seq_length - seq_length..,
            //             )),
            //         })
            //         .collect();
            //     let simple_logits = lm_logits.i(current_batch..current_batch + mini_batch_size);
            //     rq.send((simple_logits, past)).unwrap();
            //     current_batch += mini_batch_size;
            // }
        }
    } else {
    }
}
