#k-mer

# sequence dir
seq="./seq"

# k-mer length
k=11

# outdir
output="./${k}mer_output"

# list file
list="sau_list.txt"


if [ ! -d ${output} ]
then
mkdir ${output}
fi

N=$1
for name in $(cat ${list} | tail -n +$1 | head -n $[$2-$1+1])
do
                echo ${name}
                fastp                   \
                -i ${seq}/${name}/*[rR]1.fq.gz                  \
                -o ${output}/${name}_clean_1.fq.gz              \
                -I ${seq}/${name}/*[rR]2.fq.gz                  \
                -O ${output}/${name}_clean_2.fq.gz              \
                -j ${output}/${name}.json                               \
                -h ${output}/${name}.html                               \
                -5 -3
                echo "$N  ${name} ===== fastp already done  $(date +"%Y-%m-%d %H:%M:%S")"	

                kmc -k$k -m24  ${output}/${name}_clean_1.fq.gz  ${output}/${name}_out1 ${output}/ 
                kmc -k$k -m24  ${output}/${name}_clean_2.fq.gz  ${output}/${name}_out2 ${output}/ 
                kmc_tools simple ${output}/${name}_out1 ${output}/${name}_out2 intersect ${output}/${name}_inter 
                kmc_dump ${output}/${name}_inter ${output}/${k}mers-${name}.txt
                rm ${output}/${name}*            
                echo "$N  ${name} ===== ${k}-mer already done  $(date +"%Y-%m-%d %H:%M:%S")"
            	N=$[$N+1]

done

# List the kmer calculations of those strains that have been completed
for i in $(ls ./${output})
do
name=$(echo $i|sed -n -e "s/${k}mers-//g" -e 's/.txt//gp')
echo "$name    $(date +"%Y-%m-%d %H:%M:%S")" >> done_file_${k}-mer.txt
done 

# collect to one file--rbind
for name in $(cat ${list} | tail -n +$1 | head -n $[$2-$1+1])
do
    cat ${output}/${k}mers-${name}.txt >> ${output}/long-merge-kmer-file
done