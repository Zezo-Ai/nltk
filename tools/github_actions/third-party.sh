#!/usr/bin/env bash
# This install script is used in our GitHub Actions CI.
# See .github/workflows/ci.yml

set -e

# Use the THIRD_PARTY_DIR environment variable set by the workflow.
if [ -z "$THIRD_PARTY_DIR" ]; then
    echo "Error: THIRD_PARTY_DIR environment variable is not set."
    exit 1
fi

# Installing the third-party software and the appropriate env variables.
pushd "$THIRD_PARTY_DIR" 2>/dev/null || { mkdir -p "$THIRD_PARTY_DIR"; pushd "$THIRD_PARTY_DIR"; }

# Download nltk stanford dependencies
# Downloaded to $THIRD_PARTY_DIR/stanford-corenlp
# stanford_corenlp_package_zip_name=$(curl -s 'https://stanfordnlp.github.io/CoreNLP/' | grep -o 'stanford-corenlp-full-.*\.zip' | head -n1)
stanford_corenlp_package_zip_name="stanford-corenlp-4.5.1.zip"
[[ ${stanford_corenlp_package_zip_name} =~ (.+)\.zip ]]
stanford_corenlp_package_name=${BASH_REMATCH[1]}
if [[ ! -d "${stanford_corenlp_package_name}" ]]; then
    curl -L "https://nlp.stanford.edu/software/$stanford_corenlp_package_zip_name" -o "${stanford_corenlp_package_zip_name}"
    # wget -nv "https://nlp.stanford.edu/software/$stanford_corenlp_package_zip_name"
    unzip -q "${stanford_corenlp_package_zip_name}"
    rm "${stanford_corenlp_package_zip_name}"
    mv "${stanford_corenlp_package_name}" 'stanford-corenlp'
fi

# Downloaded to $THIRD_PARTY_DIR/stanford-parser
# stanford_parser_package_zip_name=$(curl -s 'https://nlp.stanford.edu/software/lex-parser.shtml' | grep -o 'stanford-parser-full-.*\.zip' | head -n1)
stanford_parser_package_zip_name="stanford-parser-full-2018-10-17.zip"
[[ ${stanford_parser_package_zip_name} =~ (.+)\.zip ]]
stanford_parser_package_name=${BASH_REMATCH[1]}
if [[ ! -d "${stanford_parser_package_name}" ]]; then
    curl -L "https://nlp.stanford.edu/software/$stanford_parser_package_zip_name" -o "${stanford_parser_package_zip_name}"
    # wget -nv "https://nlp.stanford.edu/software/$stanford_parser_package_zip_name"
    unzip -q "${stanford_parser_package_zip_name}"
    rm "${stanford_parser_package_zip_name}"
    mv "${stanford_parser_package_name}" 'stanford-parser'
fi

# Downloaded to $THIRD_PARTY_DIR/stanford-postagger
# stanford_tagger_package_zip_name=$(curl -s 'https://nlp.stanford.edu/software/tagger.shtml' | grep -o 'stanford-postagger-full-.*\.zip' | head -n1)
stanford_tagger_package_zip_name="stanford-postagger-full-2018-10-16.zip"
[[ ${stanford_tagger_package_zip_name} =~ (.+)\.zip ]]
stanford_tagger_package_name=${BASH_REMATCH[1]}
if [[ ! -d "${stanford_tagger_package_name}" ]]; then
    curl -L "https://nlp.stanford.edu/software/$stanford_tagger_package_zip_name" -o "${stanford_tagger_package_zip_name}"
    # wget -nv "https://nlp.stanford.edu/software/$stanford_tagger_package_zip_name"
    unzip -q "${stanford_tagger_package_zip_name}"
    rm "${stanford_tagger_package_zip_name}"
    mv "${stanford_tagger_package_name}" 'stanford-postagger'
fi

# Download SENNA to $THIRD_PARTY_DIR/senna
# senna_file_name=$(curl -s 'https://ronan.collobert.com/senna/download.html' | grep -o 'senna-v.*.tgz' | head -n1)
if [[ "$(uname -s)" == "Linux" ]]; then
    senna_file_name=$(curl -s 'https://ronan.collobert.com/senna/download.html' | grep -o 'senna-v.*.tgz' | head -n1)
    senna_folder_name='senna'
    if [[ ! -d "${senna_folder_name}" ]]; then
        curl -L "https://ronan.collobert.com/senna/$senna_file_name" -o "${senna_file_name}"
        # wget -nv "https://ronan.collobert.com/senna/$senna_file_name"
        tar -xzf "${senna_file_name}"
        rm "${senna_file_name}"
    fi
else
    echo "Skipping SENNA download on $(uname -s)"
fi

# Download PROVER9 to $THIRD_PARTY_DIR/prover9
if [[ "$(uname -s)" == "Linux" ]]; then
    prover9_file_name="p9m4-v05.tar.gz"
    [[ ${prover9_file_name} =~ (.+)\.tar\.gz ]]
    prover9_folder_name=${BASH_REMATCH[1]}
    if [[ ! -d "${prover9_folder_name}" ]]; then
        curl -L "https://www.cs.unm.edu/~mccune/prover9/gui/$prover9_file_name" -o "${prover9_file_name}"
        tar -xzf "${prover9_file_name}"
        mv "${prover9_folder_name}" 'prover9'
        rm "${prover9_file_name}"
    fi
else
    echo "Skipping PROVER9 download on $(uname -s)"
fi

# Download MEGAM to $THIRD_PARTY_DIR/megam
if [[ "$(uname -s)" == "Linux" ]]; then
    megam_file_name="megam_i686.opt.gz"
    [[ ${megam_file_name} =~ (.+)\.gz ]]
    megam_folder_name=${BASH_REMATCH[1]}
    if [[ ! -d "${megam_folder_name}" ]]; then
        curl -L "http://hal3.name/megam/$megam_file_name" -o "${megam_file_name}"
        gunzip -vf "${megam_file_name}"
        mkdir -p "megam"
        mv "${megam_folder_name}" "megam/${megam_folder_name}"
        chmod -R 711 "megam/${megam_folder_name}"
    fi
else
    echo "Skipping MEGAM download on $(uname -s)"
fi

# TADM requires `libtaopetsc.so` from PETSc v2.3.3, and likely has more
# tricky to install requirements, so we don't run tests for it.
# Download TADM to $THIRD_PARTY_DIR/tadm
# tadm_file_name="tadm-0.9.8.tgz"
# [[ ${tadm_file_name} =~ (.+)\.tgz ]]
# tadm_folder_name=${BASH_REMATCH[1]}
# if [[ ! -d "${tadm_folder_name}" ]]; then
#     curl -L "https://master.dl.sourceforge.net/project/tadm/tadm/tadm%200.9.8/$tadm_file_name?viasf=1" -o "${tadm_file_name}"
#     tar -xvzf "${tadm_file_name}"
#     rm "${tadm_file_name}"
#     chmod -R 711 "./tadm/bin/tadm"
# fi

# Download MaltParser to $THIRD_PARTY_DIR/maltparser
# malt_file_name=$(curl -s 'http://maltparser.org/download.html' | grep -o 'maltparser-.*\.tar.gz' | head -n1)
malt_file_name="maltparser-1.9.2.tar.gz"
[[ ${malt_file_name} =~ (.+)\.tar\.gz ]]
malt_folder_name=${BASH_REMATCH[1]}
if [[ ! -d "${malt_folder_name}" ]]; then
    curl -L "http://maltparser.org/dist/$malt_file_name" -o "${malt_file_name}"
    tar -xzf "${malt_file_name}"
    mv "${malt_folder_name}" 'maltparser'
    rm "${malt_file_name}"
fi

ls -l "$THIRD_PARTY_DIR"

popd
