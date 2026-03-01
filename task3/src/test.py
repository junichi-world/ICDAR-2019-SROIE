import argparse
import os
import torch
from my_data import MyDataset, VOCAB
from my_models import MyModel0
from my_utils import pred_to_dict
import json


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", default="cpu")
    parser.add_argument("-i", "--hidden-size", type=int, default=256)

    args = parser.parse_args()
    args.device = torch.device(args.device)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(project_root, "data")
    results_dir = os.path.join(project_root, "results")
    model_path = os.path.join(project_root, "model.pth")

    os.makedirs(results_dir, exist_ok=True)

    model = MyModel0(len(VOCAB), 16, args.hidden_size).to(args.device)
    dataset = MyDataset(None, args.device, test_path=os.path.join(data_dir, "test_dict.pth"))

    model.load_state_dict(torch.load(model_path, map_location=args.device))

    model.eval()
    with torch.no_grad():
        for key in dataset.test_dict.keys():
            text_tensor = dataset.get_test_data(key)

            oupt = model(text_tensor)
            prob = torch.nn.functional.softmax(oupt, dim=2)
            prob, pred = torch.max(prob, dim=2)

            prob = prob.squeeze().cpu().numpy()
            pred = pred.squeeze().cpu().numpy()

            real_text = dataset.test_dict[key]
            result = pred_to_dict(real_text, pred, prob)

            out_path = os.path.join(results_dir, key + ".json")
            with open(out_path, "w", encoding="utf-8") as json_opened:
                json.dump(result, json_opened, indent=4)

            print(key)


if __name__ == "__main__":
    test()
