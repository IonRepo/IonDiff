a="/home/cibran/Desktop/Data-m3gnet/"
for b in $a*/; do
    for c in $b*/; do
        for d in $c*/; do
            for e in $d*/; do
                for f in $e*; do
                    echo $f
                    python3 cli.py identify_diffusion --MD_path $f
                done
            done
        done
    done
done
