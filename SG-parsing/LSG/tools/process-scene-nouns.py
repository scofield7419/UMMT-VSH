import sys
import spacy

nlp = spacy.load('en')

extra_nouns = ['cabin', 'airport', 'terminal', 'arcade', 'park', 'apartment', 'gallery', 'school', 'studio', 'loft', 'field', 'factory', 'showroom', 'bank', 'banquet', 'court', 'salon', 'laboratory', 'station', 'store', 'lab', 'room', 'conference', 'dorm', 'lobby', 'entrance', 'restaurant', 'market', 'office', 'theater', 'skating', 'jail', 'kindergarden', 'dock', 'gym', 'cubicles', 'residential', 'mall', 'resort', 'hole', 'hostel']


def main():
    nouns = set()
    for line in sys.stdin:
        line, _ = line.split(' ')
        line = line.split('/')[2:]
        for x in line:
            parts = x.split('_')
            nouns.add(' '.join(parts))

    nouns.update(set(extra_nouns))

    for n in sorted(nouns):
        print(n)


if __name__ == '__main__':
    main()

