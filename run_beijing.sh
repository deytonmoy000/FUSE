nNodes=$2   
nThreads=$3
pyfile=$1

# alg=$1
declare -a algs=("TSMTC" "REASSIGN" "SDPR" "AGD" "fuse_test") # ("fuse_test_wo12" "fuse_test_wo34" "fuse_test_wo5") # 
declare -a tfs=("15" "30" "60") # ("20" "40" "60")
declare -a grids=("1000" "500")
declare -a seeds=("42") # "14" "25" "8" "35") # Please use any seed value as you prefer
declare -a nDivs=("8" "4" "2" "1")



# Perform the multiplication and store the result in a new variable
result=$(( variable * constant ))
for g in ${!grids[@]};
do
	for j in ${!seeds[@]};
	do
		
		for i in ${!nDivs[@]};
		do
			nDiv=${nDivs[$i]}
			seed=${seeds[$j]}
			
			for k in ${!tfs[@]};
			do
				for a in ${!algs[@]};
				do
					gs=${grids[$g]}
					
					alg=${algs[$a]}
					tf=${tfs[$k]}

					cmd0="mkdir -p result_paper_beijing/GS_${gs}m/TF_${tf}/Div_${nDiv}/Rep_${seed}"
					echo $cmd0
					$cmd0

					cmd1="mkdir -p figs_paper_beijing/GS_${gs}m/TF_${tf}/Div_${nDiv}/Rep_${seed}/${alg}"
					echo $cmd1
					$cmd1

					cmd="python main_beijing.py ${nDiv} ${tf} ${alg} ${seed} ${gs}"
					echo $cmd
					$cmd
				
				done
			done

			echo "Result generated for seed = '${seed}'"
		done
	done
done