// local model_name = "facebook/bart-large";
local model_name = "facebook/bart-large-cnn";

// local train_data = "/gds/xshen/projdata11/researchPJ/ot_abs/toy_set_discrete_files/v1/train";
// local dev_data = "/gds/xshen/projdata11/researchPJ/ot_abs/toy_set_discrete_files/v1/val";
local train_data = "/gds/xshen/projdata17/researchPJ/processed_gov/train";
local dev_data = "/gds/xshen/projdata17/researchPJ/processed_gov/val";

{
    "train_data_path": train_data,
    // "validation_data_path": dev_data,
    "dataset_reader": {
        "type": "pubmed_arxiv",  // cnn_dm
        "source_tokenizer": {
            "type": "pretrained_transformer",
            "model_name": model_name
        },
        "source_token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": model_name,
                "namespace": "tokens",
            }
        },
        "reader_settings": {
            // "source_max_tokens": 1022, // for cnndm
            // "target_max_tokens": 54, // for cnndm
            // "max_instances": 1000, // DEBUG setting
            // new 4 ot:
            "max_len_one_section_src": 500,  // 500 4 pubmed; 600 4 arxiv
            "max_len_one_sentence_abst": 40,
            "max_src_sections_nums": 25,
            "max_abst_sentences_nums": 30,
        },
    },
    "model": {
        "type": "bart_uot",
        "model_name": model_name,
        "beam_search": {
            "max_steps": 300,
            "beam_size": 4,
        },
        "model_settings": {
            "max_abst_sentences_nums": 30,
            "recombined_max_len_abst_one_sec": 200,  // 80 for pubmed; 140 for arxiv; for gov_report;
            "ot_matcher_input_size": 1024,
        },
    },
    "data_loader": {
        "batch_size": 1, // !!!!! batch_size must fixed to 1
        "shuffle": true
    },
    "trainer": {
        "cuda_device": 1,  // new
        "num_epochs": 12, // original: 3
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 3e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "correct_bias": true,
        },
        "learning_rate_scheduler": {
            "type": "polynomial_decay",
        },
        "grad_norm": 1.0,
        "run_confidence_checks": false, // new
    }
}

/*
local data_base_url = "https://storage.googleapis.com/allennlp-public-data/cnndm-combined-data-2020.07.13.tar.gz";
local train_data = data_base_url + "!cnndm-combined-data-2020.07.13/url_lists/all_train.txt";
local dev_data = data_base_url + "!cnndm-combined-data-2020.07.13/url_lists/all_val.txt";
local data_base_folder = "processed_arxiv";  // processed_arxiv processed_pubmed

plan 1 :
    "ot_matcher_epsilon": 0.006,
    "ot_matcher_tau_sinkhorn": 0.03,
plan 2 :
    "ot_matcher_epsilon": 0.05,
    "ot_matcher_tau_sinkhorn": 0.3,
*/

