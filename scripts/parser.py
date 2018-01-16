def give_ground_truth():
    with open('./ground_truth.txt') as f:
        parsedValue =  {}
        data = f.readlines()

        for _, line in enumerate(data):
                line = line.split(' ')
                words = line[-1].split('|')
                words[-1] = words[-1].replace('\n','')
                parsedValue[line[0]] = ' '.join(words)

        f.close()

    return parsedValue

print(give_ground_truth())
