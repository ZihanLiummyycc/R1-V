import numpy as np
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import cv2
import os
from PIL import Image
from object_detection.yolo.load_ckpt import load_ckpt
from object_detection.yolo.yolo_inference import YOLOInferenceAPI
from object_detection.opencv_template_matching.opencv_inference import TemplateMatchingAPI
import torch
# from object_detection.action_space import all_env_action_spaces
import tqdm

env_list = [
    # 'Riverraid', 'YarsRevenge', 'AirRaid', 'Assault', 'BattleZone',
    #     'Riverraid',
    #     'AirRaid',    'SpaceInvaders', 'BeamRider', 'Carnival', 'ChopperCommand', 'DemonAttack', 'Phoenix', 'StarGunner',
        # 'Seaquest'
        # "Tennis",
        # "Bowling"
        # "Boxing",
        # "IceHockey",
        # "Skiing",
        # "FishingDerby",
        #"DoubleDunk",
        #"Hero"
        #"ElevatorAction"
        #"Tutankham"
        #"MontezumaRevenge"
        #"Berzerk"
        #"Asteroids"
        #"Atlantis"
        #"BankHeist"
        #"WizardOfWor"
        #"Jamesbond"
        #"Enduro"
        #"Venture"
        #"TimePilot"
        #"UpNDown"
        #"Pong"
        #"PrivateEye"
        #"Qbert"
        #"Frostbite"
        #"Zaxxon"
        #"NameThisGame"
        #"Robotank"
        #"Seaquest"
        "Carnival"
        ]

for env_id in range(len(env_list)):
    game_name = env_list[env_id] + "NoFrameskip-v4"

    short_name = game_name.split("NoFrameskip-v4")[0]
    saved_path = f"images/{short_name}"
    os.makedirs(saved_path, exist_ok=True)

    print("Initializing environment...")
    env = make_atari_env(game_name, n_envs=1, seed=42)
    env = VecFrameStack(env, n_stack=4)

    obs = env.reset()
    action_space = env.envs[0].unwrapped.get_action_meanings()

    if short_name == "AirRaid":
        print("AirRaid env needs the YOLO to detect the buildings")
        yolo_model = YOLOInferenceAPI("object_detection/yolo/weight/airraid_buildings.pt")
    policy = load_ckpt(game_name, env.action_space.n)

    # cv_predictor = TemplateMatchingAPI(
    #             templates_json="object_detection/opencv_template_matching/templates.json",
    #             templates_images_dir="object_detection/opencv_template_matching/template_imgs/" + short_name.lower(),
    #             game_config_yaml="object_detection/opencv_template_matching/game_configs/" + short_name.lower() + ".yaml"
    #         )

    total_steps = 10000 # 需要执行的总步数
    sample_rate = 200
    0  # 每隔多少步采样一次
    prediction_mode = False  # 是否使用预测模式。如果为True，保存labelled images；如果为False，则只保存原始image。

    # for index in range(total_steps):
    for index in tqdm.tqdm(range(total_steps), desc=f"Processing {short_name}"):
        img = env.render(mode='rgb_array')
        ori_img = img.copy()

        obs = torch.from_numpy(obs).byte()
        obs = obs.permute(0, 3, 1, 2)  # 改成 [1, 4, 84, 84]
        action, _ = policy(obs)

        obs, reward, done, info = env.step([action])
        if index % sample_rate == 0:
            print(f"\n执行动作: {action}/{action_space[action]}, 奖励: {reward}, 是否完成: {done}")

            img = env.render(mode='rgb_array')

            if not prediction_mode:
                # Prediction mode is off, using the current image.just save raw image for training
                saved_img = Image.fromarray(img)
                saved_img.save(f"{saved_path}/raw_image_{index}.png")
                continue
            

            objects = []
            if short_name == "AirRaid":
                labeled_img, detections = yolo_model.predict_image(img, return_detections=True)
                for info in detections:
                    class_name = info["class_name"]
                    # print(class_name)
                    if class_name == "buildings":
                        top_left_position = (int(info["bbox"][0]), int(info["bbox"][1]))
                        bottom_right_position = (int(info["bbox"][2]), int(info["bbox"][3]))
                        objects.append({
                            "class": class_name,
                            "top_left_position": top_left_position,
                            "bottom_right_position": bottom_right_position,
                        })
                        print(f"class {class_name} is at {top_left_position}-{bottom_right_position}")

            img_cv2 = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)
            # labelled_img,infos = cv_predictor.predict(img_cv2,return_detections=True)

            # label_infos = infos["class_labels"]
            # bar_infos = infos["bar_types"]
            # count_infos = infos["count_types"]

            # for count_info in count_infos:
            #     print("The count for "+count_info['class_name']+f" is {count_info['count']}.")

            # if len(bar_infos) != 0:
            #     for bar_info in bar_infos:
            #         # print(bar_info)
            #         print(bar_info["description"]+f" The bar length is {bar_info['length']}.")

            # for info in label_infos:
            #     class_name = info['class_name']

            #     # # for testing
            #     # if class_name != "player":
            #     #     continue

            #     if class_name != "building":
            #         top_left_position = (int(info['position'][0]), int(info['position'][1]))
            #         height = info['rec_height']
            #         width = info['rec_width']
            #         bottom_right_position = (int(info['position'][0] + width), int(info['position'][1] + height))
            #         objects.append({
            #             "class": class_name,
            #             "top_left_position": top_left_position,
            #             "bottom_right_position": bottom_right_position,
            #         })
            #         print(f"class {class_name} is at {top_left_position}-{bottom_right_position}")
            # print (f"Step{index}: {len(label_infos)} labels plotted")

            # # save original image
            # output_path = f"{saved_path}/output_test_{index}_orig.png"
            # saved_img = Image.fromarray(ori_img)
            # saved_img.save(output_path)


            for object in objects:
                cv2.rectangle(ori_img, (object["top_left_position"][0], object["top_left_position"][1]), (object["bottom_right_position"][0], object["bottom_right_position"][1]), (0,255,0), 2)

                # font = cv2.FONT_HERSHEY_SIMPLEX
                # font_scale = 0.25
                # font_color = (0, 0, 0)  # 蓝色字体
                # thickness = 1
                # (w, h), _ = cv2.getTextSize(object['class'], font, font_scale, thickness)
                # cv2.rectangle(ori_img, (object["top_left_position"][0], object["top_left_position"][1] - h - 10), (object["top_left_position"][0] + w, object["top_left_position"][1]), (255,255,255), -1)
                # cv2.putText(ori_img, object['class'], (object["top_left_position"][0], object["top_left_position"][1] - 5), font, font_scale, font_color, thickness)

            output_path = f"{saved_path}/output_test_{index}.png"
            saved_img = Image.fromarray(ori_img)
            saved_img.save(output_path)
        if done:
            env = make_atari_env(game_name, n_envs=1, seed=42)
            env = VecFrameStack(env, n_stack=4)

            obs = env.reset()