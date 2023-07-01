import argparse
import csv


def main(args: argparse.ArgumentParser):
    with open(args.full, "r") as full_file, open(args.split, "r") as split_file:
        words_per_sentence, sentences = [], []
        for i, *_ in csv.reader(full_file, delimiter="\t", quoting=csv.QUOTE_NONE):
            words_per_sentence.append(len(i.split(' ')))
            sentences.append(i)

        predictions = [o for _, o in csv.reader(split_file, delimiter="\t", quoting=csv.QUOTE_NONE)]
        glued_predictions = []
        start = 0
        for i in range(len(words_per_sentence)):
            end = start + words_per_sentence[i]
            glued_predictions.append(predictions[start:end])
            start += words_per_sentence[i]

        assert len(glued_predictions) == len(sentences)
    
    with open(args.output, 'w') as out_file:
        for i in range(len(sentences)):
            out_file.write(f"{sentences[i]}\t{' '.join(glued_predictions[i])}\n")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--full", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()
    main(args)

