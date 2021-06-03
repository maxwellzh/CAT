import codecs
import sys

def load_data(fin):
    with codecs.open(fin, "r", encoding = "utf8") as fp:
        data = fp.readlines()
        return data

def write2file(fout, data):
    with codecs.open(fout, "w", encoding = "utf8") as fp:
        for line in data:
            fp.write(line)

def main(fin, fout):
    data = load_data(fin)
    new_data = []
    for line in data:
        line = line.strip().split()
        text = "".join(line[1:]).replace("▁", " ")
        new_line = f"{line[0]} {text}\n"
        new_data.append(new_line)
    write2file(fout, new_data)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usgae: {sys.argv[0]} bpe.txt wrd.txt")
        exit(0)
    main(sys.argv[1], sys.argv[2])
