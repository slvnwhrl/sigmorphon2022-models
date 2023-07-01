import argparse
import errno
import json
import logging
import os

import torch

from trans import transducer
from trans import vocabulary
from trans import sed
from trans import optimal_expert_substitutions
from trans import utils
from trans.train import decode


def main(args: argparse.Namespace):
    # first check if all paths exists so time is saved if not
    model_path = os.path.join(args.model_folder, "best.model")
    config_path = os.path.join(args.model_folder, "config.json")
    sed_params_path = os.path.join(args.model_folder, "sed.pkl")
    voc_path = os.path.join(args.model_folder, "vocabulary.pkl")

    for path in (args.test, args.model_folder, args.output,
                 model_path, config_path, sed_params_path, voc_path):
        if not os.path.exists(path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

    # load vocabulary
    if args.features:
        vocabulary_ = vocabulary.FeatureVocabularies.from_pickle(voc_path)
    else:
        vocabulary_ = vocabulary.Vocabularies.from_pickle(voc_path)

    # load expert
    sed_aligner = sed.StochasticEditDistance.from_pickle(sed_params_path)
    expert = optimal_expert_substitutions.OptimalSubstitutionExpert(sed_aligner)

    # load model
    # first add config to args
    logging.info("Loading model config parameters from file: %s", config_path)
    with open(config_path, "r") as c:
        model_config: dict = json.load(c)
    args_dict = vars(args)
    for key, value in model_config.items():
        args_dict[key.replace("-", "_")] = value
    transducer_ = transducer.Transducer(vocabulary_, expert, args)
    logging.info("Loading model state dict from file: %s", model_path)
    transducer_.load_state_dict(torch.load(model_path, map_location=args.device))
    transducer_.eval()

    # load data
    test_data = utils.Dataset()
    with utils.OpenNormalize(args.test, args.nfd) as f:
        for line in f:
            if args.features:
                input_, optional_target, features = line.rstrip().split(
                    "\t", 2)
                encoded_features = torch.tensor(
                    vocabulary_.encode_unseen_features(features),
                    device=args.device,
                )
                target = optional_target if optional_target else None
            else:
                input_, *optional_target = line.rstrip().split("\t", 1)
                features = encoded_features = None
                target = optional_target[0] if optional_target else None

            encoded_input = torch.tensor(vocabulary_.encode_unseen_input(input_),
                                         device=args.device)
            sample = utils.Sample(
                input_, target, encoded_input,
                features=features,
                encoded_features=encoded_features,
            )
            test_data.add_samples(sample)
    test_data_loader = test_data.get_data_loader(batch_size=args.batch_size,
                                                 device=args.device)

    # predict
    with torch.no_grad():
        dec_results = []
        if args.beam_width > 0:
            logging.info("Predicting using beam search (beam width %d)...", args.beam_width)
            with utils.Timer():
                beam_predictions = decode(transducer_, test_data_loader, args.beam_width).predictions
            dec_results.append((f"beam{args.beam_width}", beam_predictions))

        logging.info("Predicting using greedy decoding...")
        with utils.Timer():
            greedy_predictions = decode(transducer_, test_data_loader).predictions
        dec_results.append(("greedy", greedy_predictions))

        for dec_type, predictions in dec_results:
            predictions_tsv = os.path.join(args.output, f"test_{dec_type}.predictions")
            with utils.OpenNormalize(predictions_tsv, args.nfd, mode="w") as w:
                w.write("\n".join(predictions))

    logging.info("Finished.")


def cli_main():
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Train a g2p neural transducer.")

    parser.add_argument("--model-folder", type=str, required=True,
                        help="Path to the model.")
    parser.add_argument("--test", type=str, required=True,
                        help="Path to test file.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output directory.")
    parser.add_argument("--features", action="store_true", default=False,
                        help="Whether the model uses features.")
    parser.add_argument("--nfd", action="store_true", default=True,
                        help="Load data NFD-normalized. Write out in NFC.")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Batch size for evaluation.")
    parser.add_argument("--beam-width", type=int, default=-1,
                        help="Beam width for beam search decoding. "
                             "A value < 1 will disable beam search decoding (default).")
    parser.add_argument("--device", type=str, default='cpu',
                        help="Device to run inference on.")

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
