#!/bin/bash

# This script prepares the phoneme-based lexicon. It also generates the list of lexicon units
# and represents the lexicon using the indices of the units. 

srcdir=data/train
dir=data/dict_phn
mkdir -p $dir
srcdict=$1
[ -f path.sh ] && . ./path.sh

[ ! -f "$srcdict" ] && echo "No such file $srcdict" && exit 1;

# Raw dictionary preparation
cat $srcdict | grep -v "!SIL" | \
  perl -e 'while(<>){@A = split; if(! $seen{$A[0]}) {$seen{$A[0]} = 1; print $_;}}' \
  > $dir/lexicon_raw1.txt || exit 1;

sed  -e 's/[ ]*$//g' $dir/lexicon_raw1.txt > $dir/lexicon_raw.txt

dos2unix $dir/lexicon_raw.txt
# Get the set of lexicon units without noises
cut -d'	' -f2- $dir/lexicon_raw.txt | tr ' ' '\n' | sort -u   > $dir/units_raw.txt

(echo '<SPOKEN_NOISE> <SPN>'; echo '<UNK> <SPN>'; echo '<NOISE> <NSN>'; ) | \
 cat - $dir/lexicon_raw.txt | sort | uniq > $dir/lexicon.txt || exit 1;

dos2unix $dir/lexicon.txt
(echo '<NSN>'; echo '<SPN>';) | cat - $dir/units_raw.txt | awk '{print $1 " " NR}' > $dir/units.txt

# Convert phoneme sequences into the corresponding sequences of units indices, encoded by units.txt
utils/sym2int.pl -f 2- $dir/units.txt < $dir/lexicon.txt > $dir/lexicon_numbers.txt

echo "Phoneme-based dictionary preparation succeeded"
