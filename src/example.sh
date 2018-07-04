rm -f out/result.txt
python main_single.py -d lsac -e 0.0 | grep "###Result" >> out/result.txt
python main_single.py -d lsac -e 0.05 | grep "###Result" >> out/result.txt
python main_single.py -d lsac -e 0.1 | grep "###Result" >> out/result.txt
python main_single.py -d lsac -e 0.2 | grep "###Result" >> out/result.txt
python main_single.py -d lsac -e 0.4 | grep "###Result" >> out/result.txt
python sampleplot.py out/result.txt lsac
