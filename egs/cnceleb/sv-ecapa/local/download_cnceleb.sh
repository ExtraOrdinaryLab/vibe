#!/bin/bash

cnceleb1_dir=/home/jovyan/corpus/audio/cnceleb1
cnceleb2_dir=/home/jovyan/corpus/audio/cnceleb2

[ ! -d ${download_dir} ] && mkdir -p ${download_dir}

wget https://openslr.elda.org/resources/82/cn-celeb_v2.tar.gz -P ${cnceleb1_dir}
tar -xzvf ${download_dir}/cn-celeb_v2.tar.gz -C ${cnceleb1_dir}

wget https://openslr.elda.org/resources/82/cn-celeb2_v2.tar.gzaa -P ${cnceleb2_dir}
wget https://openslr.elda.org/resources/82/cn-celeb2_v2.tar.gzab -P ${cnceleb2_dir}
wget https://openslr.elda.org/resources/82/cn-celeb2_v2.tar.gzac -P ${cnceleb2_dir}
cat ${cnceleb2_dir}/cn-celeb2_v2.tar.gza* >${cnceleb2_dir}/cn-celeb2_v2.tar.gz
tar -xzvf ${cnceleb2_dir}/cn-celeb2_v2.tar.gz -C ${cnceleb2_dir}