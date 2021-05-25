import codecs
import sys
import unicodedata

def load_data(fin):
    with codecs.open(fin, "r", encoding = "utf8") as fp:
        return fp.readlines()

def write2file(fout, data):
    with codecs.open(fout, "w", encoding = "utf8") as fp:
        for line in data:
            fp.write(line)

def add_lex(line_id, pie, txt, lex, pdic):

    j = 0
    for w in txt:
        i = j+1
        while i < len(pie):
            tmp = "".join(pie[j:i]).replace("▁", "")
            if tmp == w:
                break
            i += 1
        pie_seq = " ".join(pie[j:i])
        j = i
        if pie_seq == "":
            print(line_id, w, pie_seq, i, j, len(pie))
            print(pie, txt)
            continue
        if w not in lex:
            lex[w] = set()
        lex[w].add(pie_seq)
    return lex

def piece2id(fin):
    data = load_data(fin)
    dic = {}
    for i in range(len(data)):
        pie = data[i].strip().split()[0]
        dic[pie] = str(i)
    return dic

def build_lexicon(fpie, fin, fphn, unk_id, des):
    text_pie = load_data(fpie)
    text = load_data(fin)
    pdic = piece2id(fphn)

    lex = {}
    assert len(text_pie) == len(text)
    for i in range(len(text_pie)):
        line_pie = text_pie[i]
        line_pie = unicodedata.normalize("NFKC", line_pie)
        line_pie = line_pie.strip().split()[1:]
        line_txt = text[i]
        line_txt = unicodedata.normalize("NFKC", line_txt)
        line_txt = line_txt.strip().split()[1:]
        lex = add_lex(i, line_pie, line_txt, lex, pdic)
    
    fout = f"{des}/lexicon_tmp.txt"
    data = []
    for w in lex:
        pie_seq = list(lex[w])[-1]
        line = f"{w} {pie_seq}\n"
        data.append(line)
    write2file(fout, data)

    fout = f"{des}/lexicon.txt"
    data = []
    for w in lex:
        pie_seq = list(lex[w])
        assert len(pie_seq) == 1
        id_seq = []
        for p in pie_seq[-1].split():
            pid = pdic[p] if p in pdic else unk_id
            id_seq.append(pid)
        if unk_id in id_seq:
            continue
        id_seq = " ".join(id_seq)
        line = f"{w} {id_seq}\n"
        data.append(line)
    write2file(fout, data)

if __name__ == "__main__":
    
    if len(sys.argv) != 6:
        print(f"Usage: {sys.argv[0]} text.piece text phones <unkid> des")
        exit(0)
    '''
    cur = "/mnt/nvme_workspace/pengwenjie/CAT/egs/commonvoice/v12"
    f1 = f"{cur}/tmp/text.piece"
    f2 = f"{cur}/tmp/text_pos"
    f3 = f"{cur}/data/local/dict_bpe/train_bpe150.vocab"
    f4 = f"{cur}/2"
    f5 = f"{cur}/12"
    '''
    build_lexicon(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    #build_lexicon(f1, f2, f3, f4, f5)
