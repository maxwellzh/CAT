
. ./cmd.sh
. ./path.sh

nbpe=$1
bpemode=$2
textdir=$3
train_set=$4
dev_set=$5
test_set=$6

dir=data/local/dict_bpe
mkdir -p $dir
bpemodel=$dir/train_bpe${nbpe}
mkdir -p $textdir

unk_id=2

set -euo pipefail

# use all the text data to generate bpe units

# remove puncuations of training text
cat data/$train_set/text | cut -f 2- -d " " - | sed 's/"//g' | sed 's/,//g' | sed 's/\.//g' | sed 's/\?//g' | \
    sed 's/\!//g' | sed 's/…//g' | sed 's/;//g' | sed 's/  / /g' | sed 's/  / /g' | sed 's/ $//g' | sed "s/’/'/g" > $dir/input.txt
#
# train sentencepiece model using training text
# input: input text for training
# vocab_size: wp size. here $nbpe=150
# bos_id: id of <s>
# eos_id: id of </s>
# unk_id: id of <SPN> here $unk_id=2
# model_type: model for training spm_train, supported: [unigram, bpe, char, word]. here $bpemode=unigram
# model_prefix: prefix of the filename to save model 
# input_sentence_size: 
# unk_surface: symbol to replace unk_id when using python3 local/spm_decode.py with wp id as input
python3 local/spm_train.py --input=$dir/input.txt --vocab_size=${nbpe} --bos_id=0 --eos_id=1 --unk_id=$unk_id \
    --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000 \
    --treat_whitespace_as_suffix=false --unk_surface="<SPN>"

for set in train dev test; do
    mkdir -p $textdir/$set
    curdir=$textdir/$set
    tmp_set=`eval echo '$'${set}_set`
    echo $tmp_set
    cp data/$tmp_set/text $curdir
    cat  $curdir/text | cut -f 2- -d " " - | sed 's/"//g' | sed 's/,//g' | sed 's/\.//g' | sed 's/\?//g' | sed 's/\!//g' \
        | sed 's/…//g' | sed 's/;//g' | sed 's/  / /g' | sed 's/  / /g' | sed 's/ $//g' | sed "s/’/'/g" > $curdir/text.tmp
    awk '{print $1}' $curdir/text > $curdir/text.uttid
    paste -d " " $curdir/text.uttid $curdir/text.tmp > $curdir/text_pos
    # model: path for saved model
    # output_format: specify encoded text format, support: [id, piece]
    python3 local/spm_encode.py --model=${bpemodel}.model --output_format=id < $curdir/text.tmp > $curdir/text.id_tmp || exit 1;
    python3 local/spm_encode.py --model=${bpemodel}.model --output_format=piece < $curdir/text.tmp > $curdir/text.piece_tmp || exit 1;
    paste -d ' ' $curdir/text.uttid $curdir/text.id_tmp > $curdir/text.id
    paste -d ' ' $curdir/text.uttid $curdir/text.piece_tmp > $curdir/text.piece
    rm $curdir/text.{id_tmp,piece_tmp}
done


# <s> and </s> needs to be removed for units.txt. Note normally text.id should not contain <s> and </s>, here we explicitly handle this.
cat  $textdir/train/text.id | cut -f 2- -d " " | tr ' ' '\n' | sort -n | uniq | awk '{print $1 " " NR}' | grep -v "^0 0$" | grep -v "^1 1$" > $dir/units.txt

cat $textdir/train/text.id | cut -f 2- -d " " - > $dir/train.tmp
python3 local/spm_decode.py --model=${bpemodel}.model --input_format=id < $dir/train.tmp | tr ' ' '\n' | sort | uniq | grep -v "^$" | grep -v '\<SPN\>' > $dir/train.wrd

python3 local/spm_encode.py --model=${bpemodel}.model --output_format=id  < $dir/train.wrd | paste -d " " \
    $dir/train.wrd - > $dir/lexicon.txt || exit 1;

echo "<SPN> $unk_id" >> $dir/lexicon.txt
grep -v "<SPN>"  $dir/lexicon.txt > $dir/lexicon_raw_nosil.txt
grep -v "<SPN>" $dir/units.txt > $dir/units_nosil.txt

utils/sym2int.pl -f 2- $dir/units.txt < $dir/lexicon.txt > $dir/lexicon_numbers.txt

echo "Succeed in generating dict"
