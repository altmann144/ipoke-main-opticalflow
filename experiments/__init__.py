from experiments.first_stage_image import FirstStageImageModel
from experiments.poke_encoder import PokeEncoderModel
from experiments.poke_encoder_FC import PokeEncoderModelFC
from experiments.img_encoder import FirstStageImageModelFC
from experiments.first_stage_video import FirstStageSequenceModel
from experiments.second_stage_video import SecondStageVideoModel
from experiments.second_stage_video_fc import SecondStageVideoModelFC
from experiments.third_stage_flow_encoder_fc import ThirdStageFlowEncoderFC
from experiments.third_stage_video_fc import ThirdStageVideoModelFC




__experiments__ = {
    "img_encoder": FirstStageImageModel,
    "poke_encoder": PokeEncoderModel,
    "first_stage": FirstStageSequenceModel,
    "second_stage": SecondStageVideoModel,
    "poke_encoder_FC": PokeEncoderModelFC,
    "img_encoder_fc": FirstStageImageModelFC,
    "second_stage_fc": SecondStageVideoModelFC,
    "flow_encoder_fc": ThirdStageFlowEncoderFC,
    "third_stage_fc": ThirdStageVideoModelFC,
}


def select_experiment(config,dirs, devices):
    experiment = config["general"]["experiment"]
    model_name = config["general"]["model_name"]
    if experiment not in __experiments__:
        raise NotImplementedError(f"No such experiment! {experiment}")
    if config["general"]["restart"]:
        print(f"Restarting run \"{model_name}\" of type \"{experiment}\". Device: {devices}")
    else:
        print(f"New run \"{model_name}\" of type \"{experiment}\". Device: {devices}")
    return __experiments__[experiment](config, dirs, devices)
