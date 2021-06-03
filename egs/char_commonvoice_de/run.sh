#!/bin/bash

. ./cmd.sh
. ./path.sh

stage=8
stop_stage=8

data_de=/home/usr-M8C3WVQA/data/CommonVoice/cv-corpus-5.1-2020-06-22/de
# experiment home
DIR=`pwd`
# number of parallel jobs
nj=28

lang=de
train_set=train_"$(echo "${lang}" | tr - _)"
dev_set=dev_"$(echo "${lang}" | tr - _)"
test_set=test_"$(echo "${lang}" | tr - _)"
recog_set="${dev_set} ${test_set}"
nbpe=150
bpemode='char'
bpetext=data/bpe_text

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Data Preparation and FST Construction"
    # python3 local/resample.py --prev_tr $data_de/validated.tsv --prev_dev $data_de/dev.tsv \
    #     --to_tr $data_de/resampled_tr.tsv --to_dev $data_de/resampled_dev.tsv

    for part in "test" "dev" "validated"; do
        # use underscore-separated names in data directories.
        local/data_prep.pl $data_de ${part} data/"$(echo "${part}_${lang}" | tr - _)" || exit 1;
    done

    utils/copy_data_dir.sh data/"$(echo "validated_${lang}" | tr - _)" data/${train_set} || exit 1;
    # utils/copy_data_dir.sh data/"$(echo "resampled_tr_${lang}" | tr - _)" data/${train_set} || exit 1;
    # utils/copy_data_dir.sh data/"$(echo "resampled_dev_${lang}" | tr - _)" data/${dev_set} || exit 1;
    utils/filter_scp.pl --exclude data/${dev_set}/wav.scp data/${train_set}/wav.scp > data/${train_set}/temp_wav.scp || exit 1;
    utils/filter_scp.pl --exclude data/${test_set}/wav.scp data/${train_set}/temp_wav.scp > data/${train_set}/wav.scp || exit 1;
    utils/fix_data_dir.sh data/${train_set} || exit 1;


    echo "Dictionary and Json Data Preparation"
    local/cv_prepare_word_dict.sh $nbpe $bpemode $bpetext $train_set $dev_set $test_set || exit 1

    ctc-crf/ctc_compile_dict_token.sh --dict_type "bpe" data/local/dict_bpe \
        data/local/lang_bpe_tmp data/lang_bpe || exit 1;
    echo "Building n-gram LM model."
    
    # train.txt without uttid for training n-gramm
    cat $bpetext/train/text_pos | cut -f 2- -d " " - > data/local/dict_bpe/train.txt || exit 1;
    local/cv_train_lm.sh data/local/dict_bpe/train.txt data/local/dict_bpe/ data/local/local_lm || exit 1;
    local/cv_format_local_lms.sh --lang-suffix "bpe" || exit 1;
fi 

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Fbank Feature Generation"
    # Perturb the speaking speed to achieve data augmentation
    echo "Generate 3 way speed perturbation."
    utils/perturb_data_dir_speed.sh 0.9 data/${train_set} data/temp1 || exit 1;
    utils/perturb_data_dir_speed.sh 1.0 data/${train_set} data/temp2 || exit 1;
    utils/perturb_data_dir_speed.sh 1.1 data/${train_set} data/temp3 || exit 1;
    utils/combine_data.sh data/${train_set}_sp data/temp1 data/temp2 data/temp3 || exit 1;
    rm -r data/temp1 data/temp2 data/temp3
    if ! utils/validate_data_dir.sh --no-feats --no-text data/${train_set}_sp; then
        echo "$0: Validation failed.  If it is a sorting issue, try the option '--always-include-prefix true'."
        exit 1
    fi

    cat  data/${train_set}_sp/text | cut -f 2- -d " " - | sed 's/"//g' | sed 's/,//g' | sed 's/\.//g' | sed 's/\?//g' | sed 's/\!//g' \
        | sed 's/…//g' | sed 's/;//g' | sed 's/  / /g' | sed 's/  / /g' | sed 's/ $//g' | sed "s/’/'/g" > data/${train_set}_sp/text.tmp
    awk '{print $1}' data/${train_set}_sp/text > data/${train_set}_sp/text.uttid
    paste -d " " data/${train_set}_sp/text.uttid data/${train_set}_sp/text.tmp > data/${train_set}_sp/text_pos

    # utils/data/perturb_data_dir_speed_3way.sh data/$train_set data/${train_set}_sp || exit 1;
    # utils/data/perturb_data_dir_speed_3way.sh data/$dev_set data/${dev_set}_sp || exit 1;

    fbankdir=fbank
    for set in ${test_set} ${dev_set} ${train_set}_sp; do
        steps/make_fbank.sh --cmd "$train_cmd" --nj $nj data/$set exp/make_fbank/$set $fbankdir || exit 1;
        utils/fix_data_dir.sh data/$set || exit;
        steps/compute_cmvn_stats.sh data/$set exp/make_fbank/$set $fbankdir || exit 1;
    done
fi

train_set=${train_set}_sp
# dev_set=${dev_set}_sp

data_tr=data/$train_set
data_cv=data/$dev_set

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    # Convert word sequences to label sequences according to lexicon_numbers.txt and text files in data/lang_bpe
    # ...the result will be placed in $data_tr/ and $data_cv/
    cp $bpetext/dev/text_pos $data_cv/
    python3 ctc-crf/prep_ctc_trans.py data/lang_bpe/lexicon_numbers.txt $data_tr/text_pos \
            "<SPN>" > $data_tr/text_number || exit 1;
    python3 ctc-crf/prep_ctc_trans.py data/lang_bpe/lexicon_numbers.txt $data_cv/text_pos \
            "<SPN>" > $data_cv/text_number || exit 1;
    echo "Convert text_number finished"

    # Prepare denominator
    cat $data_tr/text_number | sort -k 2 | uniq -f 1 > $data_tr/unique_text_number || exit 1;
    mkdir -p data/den_meta
    # TODO: Huahuan 2021.6.1 maybe try --ngram-order=3 for bpe
    chain-est-phone-lm ark:$data_tr/unique_text_number data/den_meta/phone_lm.fst || exit 1;    
    python3 ctc-crf/ctc_token_fst_corrected.py den data/lang_bpe/tokens.txt | fstcompile \
        | fstarcsort --sort_type=olabel > data/den_meta/T_den.fst || exit 1;
    fstcompose data/den_meta/T_den.fst data/den_meta/phone_lm.fst > data/den_meta/den_lm.fst || exit 1;
    echo "Prepare denominator finished"

    # calculate and save the weight for each label sequence based on text_number and phone_lm.fst
    path_weight $data_tr/text_number data/den_meta/phone_lm.fst > $data_tr/weight || exit 1;
    path_weight $data_cv/text_number data/den_meta/phone_lm.fst > $data_cv/weight || exit 1;
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


if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    echo "nn training."
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
    echo ""
fi

dir=exp/char_orinsplit_sp_conformer_Mv1
if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
    cp data/test_de/text data/test_de/text.origin
    cp $bpetext/test/text_pos data/test_de/text
    for set in test_de; do
        oldlm=tgpr
        mkdir -p $dir/decode_bd_$oldlm
        ln -snf ../logits/$set $dir/decode_bd_$oldlm/logits
        ctc-crf/decode.sh --stage 1 \
            --cmd "$decode_cmd" --nj $nj --acwt 1.0 \
            data/lang_bpe_test_bd_$oldlm data/$set data/all_ark/$set.scp $dir/decode_bd_$oldlm || exit 1

        for lm in tgconst fgconst; do
            mkdir -p $dir/decode_bd_$lm
            steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang_bpe_test_bd_{$oldlm,$lm} data/${set} $dir/decode_bd_{$oldlm,$lm} || exit 1
        done

        for lm in tg fgpr fg; do
            mkdir -p $dir/decode_bd_$lm
            steps/lmrescore.sh --cmd "$decode_cmd" --mode 3 data/lang_bpe_test_bd_{$oldlm,$lm} data/${set} $dir/decode_bd_{$oldlm,$lm} || exit 1
        done
        grep WER $dir/decode_bd_*/wer_* | utils/best_wer.sh 
        grep WER $dir/decode_bd_*/cer_* | utils/best_wer.sh 
    done
fi
