poke_embedder_models ={
'plants-ss64': {
        'ckpt': 'scratch/poke_encoder_FC/ckpt/poke_enc_fc-64/1/epoch=24-lpips-val=0.239.ckpt',
        'model_name': 'poke_enc_fc-64',
        'tgt_name': 'plants_64'
    },
}
first_stage_models = {
'plants-ss64' : {
        'ckpt': 'scratch/first_stage/ckpt/first_stage_fc-64/0/epoch=12-FVD-val=70.894.ckpt',
        'model_name': 'first_stage_fc-64',
        'tgt_name':'plants_64'
    },
'plants-ss64-128' : {
        'ckpt': 'scratch/first_stage/ckpt/plants-ss64-128/0/epoch=20-FVD-val=90.525.ckpt',
        # 'model_name': 'first_stage_fc-64',  # already in the ckpt string above
        # 'tgt_name':'plants_64'
    },
'plants-ss64-256' : {
        'ckpt': 'scratch/first_stage/ckpt/plants-ss64-256/0/epoch=20-FVD-val=88.484.ckpt',
        # 'model_name': 'first_stage_fc-64',
        # 'tgt_name':'plants_64'
    },
'old': {
        'ckpt': 'scratch/first_stage/ckpt/old/0/epoch=99-step=210999.ckpt',
        'model_name': 'first_stage_fc-64',
        'tgt_name': 'plants_64'
    },
}
conditioner_models = {
'plants-ss64': {
        'ckpt': 'scratch/img_encoder_fc/ckpt/img_enc_fc-64/3/epoch=29-lpips-val=0.069.ckpt',
        'model_name': 'img_enc_fc-64',
        'tgt_name': 'plants_64'
    },

}

flow_conditioner_models ={}