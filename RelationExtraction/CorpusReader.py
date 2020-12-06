from FeatureExtract import FeatureExtract


class CorpusReader:
    def __init__(self, filepath):
        self.file_path = filepath

    def feature_extract(self):
        textual_features = []
        try:
            print('Scanning file: ' + self.file_path)
            with open(self.file_path, 'r') as file:
                for line in file:
                    input_sentence = line
                    input_relation = next(file)
                    features = FeatureExtract(input_sentence, input_relation)
                    attrs = vars(features)
                    print('\n'.join("%s -> %s" % item for item in attrs.items()))
                    print("\n")
                    textual_features.append(features)
                    extra_line1, extra_line2 = (next(file), next(file))
        except StopIteration:
            pass
        return textual_features
