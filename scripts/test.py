def computeGroundTruthSentences():
    with open('./train_ground_truth.txt') as f:
        data = f.readlines()
        groundTruthDict = {}

        for _, line in enumerate(data):
            chunks = line.split('\|/')
            fileNumber = chunks[0]
            groundTruth = chunks[-1]

            groundTruthDict[fileNumber] = groundTruth

        ground_truth_sentences = list(map(
            lambda x: x[1],
            sorted(groundTruthDict.items(), key=lambda x: int(x[0]))
        ))

        f.close()

    return ground_truth_sentences

if __name__ == '__main__':
    ground_truth_sentences = computeGroundTruthSentences()
    print(ground_truth_sentences)
