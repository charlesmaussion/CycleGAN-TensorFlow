def give_ground_truth():
    f = open('ground_truth.txt')
    parsedValue =  {}
    data = f.readlines()
    for n, line in enumerate(data, 1):
            line = line.split(' ')
            words = line[-1].split('|')
            words[-1] = words[-1].replace('\n','')
            parsedValue[line[0]] = ' '.join(words)
    print('-----------------')
    f.close()
    return parsedValue

#print(give_ground_truth())
