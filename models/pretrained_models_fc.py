first_stage_models = {
'plants-ss64' : {'ckpt': 'scratch/first_stage/ckpt/first_stage_fc-64/0/epoch=12-FVD-val=70.894.ckpt'},
'plants-ss64-128' : {'ckpt': 'scratch/first_stage/ckpt/plants-ss64-128/0/epoch=52-FVD-val=76.555.ckpt'},
'plants-ss64-256' : {'ckpt': 'scratch/first_stage/ckpt/plants-ss64-256/0/epoch=20-FVD-val=88.484.ckpt'},
'iper-ss64-128': {'ckpt': 'scratch/first_stage/ckpt/iper-ss64-128/0/epoch=75-FVD-val=82.930.ckpt'},
'iper-ss64-256': {'ckpt': 'scratch/first_stage/ckpt/iper-ss64-256/0/epoch=39-FVD-val=96.259.ckpt'},
'iper-ss64-512': {'ckpt': 'scratch/first_stage/ckpt/iper-ss64-512/0/epoch=47-FVD-val=84.512.ckpt'},
}
poke_embedder_models = {
'plants-ss64': {'ckpt': 'scratch/poke_encoder_FC/ckpt/poke_enc_fc-64/1/epoch=24-lpips-val=0.239.ckpt'},
'iper-ss64-64':{'ckpt': 'scratch/poke_encoder_FC/ckpt/iper-ss64-64/1/epoch=27-lpips-val=0.201.ckpt'},
'iper-ss64-128':{'ckpt': 'scratch/poke_encoder_FC/ckpt/iper-ss64-128/0/epoch=25-lpips-val=0.205.ckpt'},
}
conditioner_models = {
'plants-ss64': {'ckpt': 'scratch/img_encoder_fc/ckpt/img_enc_fc-64/3/epoch=29-lpips-val=0.069.ckpt'},
'iper-ss64-64': {'ckpt': 'scratch/img_encoder_fc/ckpt/iper-ss64-64/0/epoch=47-lpips-val=0.043.ckpt'},

}
second_stage_models = {
'plants-ss64-128': {'ckpt': 'scratch/second_stage_fc/ckpt/plants-ss64-128/0/epoch=85-FVD-val=88.523.ckpt'},
'plants-ss64-128-radial': {'ckpt': 'scratch/second_stage_fc/ckpt/plants-ss64-128-radial/1/epoch=19-FVD-val=80.156.ckpt'},
'iper-ss64-128': {'ckpt': 'scratch/second_stage_fc/ckpt/iper-ss64-128/0/epoch=31-FVD-val=130.824.ckpt'},

}
flow_encoder_models = {
'plants-ss64-128': {'ckpt': 'scratch/flow_encoder_fc/ckpt/plants-ss64-128/1/epoch=19-lpips-val=0.086.ckpt'},
# 'plants-ss64-120': {'ckpt': 'scratch/flow_encoder_fc/ckpt/plants-ss64-120/1/epoch=19-lpips-val=0.086.ckpt'},
'plants-ss64-112': {'ckpt': 'scratch/flow_encoder_fc/ckpt/plants-ss64-112/0/epoch=33-lpips-val=0.082.ckpt'},
'plants-ss64-96': {'ckpt': 'scratch/flow_encoder_fc/ckpt/plants-ss64-96/0/epoch=35-lpips-val=0.093.ckpt'},
'iper-ss64-112': {'ckpt': 'scratch/flow_encoder_fc/ckpt/iper-ss64-112/0/epoch=15-lpips-val=0.040.ckpt'},

}