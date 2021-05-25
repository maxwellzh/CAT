#!/bin/bash

. ./cmd.sh
. ./path.sh

stage=2
stop_stage=5

data_de=/home/usr-M8C3WVQA/data/CommonVoice/cv-corpus-5.1-2020-06-22/de
# experiment home
DIR=`pwd`
# number of parallel jobs
nj=20

lang=de
train_set=train_"$(echo "${lang}" | tr - _)"
dev_set=dev_"$(echo "${lang}" | tr - _)"
test_set=test_"$(echo "${lang}" | tr - _)"
recog_set="${dev_set} ${test_set}"
nbpe=150
bpemode=unigram
beptext=data/bpe_text

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Data Preparation and FST Construction"
    for part in "validated" "test" "dev"; do
        # use underscore-separated names in data directories.
        local/data_prep.pl $data_de ${part} data/"$(echo "${part}_${lang}" | tr - _)" || exit 1;
    done

    utils/copy_data_dir.sh data/"$(echo "validated_${lang}" | tr - _)" data/${train_set} || exit 1;
    utils/filter_scp.pl --exclude data/${dev_set}/wav.scp data/${train_set}/wav.scp > data/${train_set}/temp_wav.scp || exit 1;
    utils/filter_scp.pl --exclude data/${test_set}/wav.scp data/${train_set}/temp_wav.scp > data/${train_set}/wav.scp || exit 1;
    utils/fix_data_dir.sh data/${train_set} || exit 1;


    echo "Dictionary and Json Data Preparation"
    local/cv_prepare_word_dict.sh $nbpe $bpemode $beptext || exit 1

    ctc-crf/ctc_compile_dict_token.sh --dict_type "bpe" data/local/dict_bpe \
        data/local/lang_bpe_tmp data/lang_bpe || exit 1;
    echo "Building n-gram LM model."
    
    # train.txt without uttid for training n-gramm
    cat $beptext/train/text_pos | cut -f 2- -d " " - > data/local/dict_bpe/train.txt || exit 1;
    local/cv_train_lm.sh data/local/dict_bpe/train.txt data/local/dict_bpe/ data/local/local_lm || exit 1;
    local/cv_format_local_lms.sh --lang-suffix "bpe" || exit 1;
fi 

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Fbank Feature Generation"
    # Perturb the speaking speed to achieve data augmentation
    # echo "Generate 3 way speed perturbation."
    # utils/perturb_data_dir_speed.sh 0.9 data/${train_set} data/temp1 || exit 1;
    # utils/perturb_data_dir_speed.sh 1.0 data/${train_set} data/temp2 || exit 1;
    # utils/perturb_data_dir_speed.sh 1.1 data/${train_set} data/temp3 || exit 1;
    # utils/combine_data.sh data/${train_set}_sp data/temp1 data/temp2 data/temp3 || exit 1;
    # rm -r data/temp1 data/temp2 data/temp3
    # if ! utils/validate_data_dir.sh --no-feats --no-text data/${train_set}_sp; then
    #     echo "$0: Validation failed.  If it is a sorting issue, try the option '--always-include-prefix true'."
    #     exit 1
    # fi
    # utils/data/perturb_data_dir_speed_3way.sh data/$train_set data/${train_set}_sp || exit 1;
    # utils/data/perturb_data_dir_speed_3way.sh data/$dev_set data/${dev_set}_sp || exit 1;

    fbankdir=fbank
    for set in ${test_set} ${dev_set} ${train_set}; do
        steps/make_fbank.sh --cmd "$train_cmd" --nj $nj data/$set exp/make_fbank/$set $fbankdir || exit 1;
        utils/fix_data_dir.sh data/$set || exit;
        steps/compute_cmvn_stats.sh data/$set exp/make_fbank/$set $fbankdir || exit 1;
    done
fi

train_set=${train_set} #_sp
# dev_set=${dev_set}_sp

data_tr=data/$train_set
data_cv=data/$dev_set

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    # Convert word sequences to label sequences according to lexicon_numbers.txt and text files in data/lang_phn
    # ...the result will be placed in $data_tr/ and $data_cv/
    for set in train dev; do
        ctc-crf/prep_ctc_trans.py data/lang_bpe/lexicon_numbers.txt $beptext/$set/text_pos \
            "<SPN>" > $beptext/$set/text_id_number || exit 1;
    done
    echo "Convert text_number finished"

    # Prepare denominator
    cat $beptext/train/text_id_number | sort -k 2 | uniq -f 1 > $beptext/train/unique_text_id_number || exit 1;
    mkdir -p data/den_meta
    chain-est-phone-lm ark:$beptext/train/unique_text_id_number data/den_meta/phone_lm.fst || exit 1;
    ctc-crf/ctc_token_fst_corrected.py den data/lang_bpe/tokens.txt | fstcompile \
        | fstarcsort --sort_type=olabel > data/den_meta/T_den.fst || exit 1;
    fstcompose data/den_meta/T_den.fst data/den_meta/phone_lm.fst > data/den_meta/den_lm.fst || exit 1;
    echo "Prepare denominator finished"

    # calculate and save the weight for each label sequence based on text_number and phone_lm.fst
    path_weight $beptext/train/text_id_number data/den_meta/phone_lm.fst > $beptext/train/weight || exit 1;
    path_weight $beptext/dev/text_id_number data/den_meta/phone_lm.fst > $beptext/dev/weight || exit 1;
    echo "Prepare weight finished"

fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    mkdir -p data/all_ark
    data_test=data/$test_set
    for set in test cv tr; do
        tmp_data=`eval echo '$'data_$set`
        feats="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$tmp_data/utt2spk scp:$tmp_data/cmvn.scp scp:$tmp_data/feats.scp ark:- |"
        ark_dir=$(readlink -f data/all_ark)/$set.ark
        copy-feats "$feats" "ark,scp:$ark_dir,data/all_ark/$set.scp" || exit 1
    done
    echo "Copy feats finished"
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    mkdir -p data/pickle
    python3 ctc-crf/convert_to.py -f=pickle -W \
        data/all_ark/cv.scp $data_cv/text_number $data_cv/weight data/pickle/cv.pickle || exit 1
    python3 ctc-crf/convert_to.py -f=pickle \
        data/all_ark/tr.scp $data_tr/text_number $data_tr/weight data/pickle/tr.pickle || exit 1
fi

exit 0

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    echo "nn training."
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
    for set in test_de; do
        CUDA_VISIBLE_DEVICES=3 \
          ctc-crf/decode.sh --cmd "$decode_cmd" --nj $nj --acwt 1.0 \
          data/lang_phn_test data/$set data/test_data/test.scp $dir/decode
    done
fi
if [ $stage -le 8 ]; then
    for set in test_de; do
        mkdir -p $dir/decode_bd_fgconst
        steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang_phn_test{,_bd_fgconst} data/${set} $dir/decode{,_bd_fgconst} || exit 1;
        mkdir -p $dir/decode_tg
        steps/lmrescore.sh --cmd "$decode_cmd" --mode 3 data/lang_phn_test{,_bd_tg} data/${set} $dir/decode{,_bd_tg} || exit 1;
    done
fi
