params = {
        "train_batch_size": 10,

        "entity_embed_size": 50,

        "word_gru_hidden_size": 300,
        "answer_gru_hidden": 300,

        # shared between input fusion GRU, question GRU,
        # attentional GRU and memory update linear layer
        "memory_size": 300,

        # attentions
        "fact_attn_hidden": 600,
        "word_attn_size": 600,

        # dropouts
        "before_answer_gru_dropout": 0.4,
        "before_dense_dropout": 0.2,
        "after_word_gru_dropout": 0.2,
        "after_fusion_gru_dropout": 0.2,
        "after_attentional_gru_dropout": 0.2,
        "after_memory_update_dropout": 0.2,
        "after_question_gru_dropout": 0.2,

        "learning_rate": 1e-3,     # Default learning rate for Adam optimizer
        "neg_pos_ratio": 9.5,
        "weight_decay": 1e-5,
        "empty_question": False,   # Change to allow empty questions
        "conv": True,              # Default to True to fine-tune the word vectors
        "num_of_pass": 1,          # Change to try different number of memory pass
}
