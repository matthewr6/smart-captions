import csv


def get_captions(filename):
    """
        Return the captions from the csv filename as a dictionary
        from index to caption as a string.
    """
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        lines = list(reader)
        lines = [(int(idx), caption) for idx, caption in lines]
    
    return dict(lines)




def main():
    caption_file = '../data/captions.csv'
    captions = get_captions(caption_file)


if __name__ == '__main__':
    main()