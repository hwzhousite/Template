#!/bin/bash
if [ $# -ne 9 ]; then
    echo "xxx.sh [数据集名称] [模型路径] [1/0是否含北交所] [模型类型字A/B/C] [mask_mf] [indexName] [t/task]"
    exit 1
fi

python -W ignore -u ./batch_test_X.py --Factor_constraint=$7 --indexTarget=$9 --minw=$8 --multitask=$5 --indexName=$6 --indexpull=$4 --DS=$1 --model=$2 --withBJ=$3 --tradeCost 0.0005 --GPU=0 --period=week --cfd=20180101 --cld=20181231 >> week_log0.log 2>&1 &
python -W ignore -u ./batch_test_X.py --Factor_constraint=$7 --indexTarget=$9 --minw=$8 --multitask=$5 --indexName=$6 --indexpull=$4 --DS=$1 --model=$2 --withBJ=$3 --tradeCost 0.0005 --GPU=1 --period=week --cfd=20190101 --cld=20191231 >> week_log1.log 2>&1 &
python -W ignore -u ./batch_test_X.py --Factor_constraint=$7 --indexTarget=$9 --minw=$8 --multitask=$5 --indexName=$6 --indexpull=$4 --DS=$1 --model=$2 --withBJ=$3 --tradeCost 0.0005 --GPU=2 --period=week --cfd=20200101 --cld=20201231 >> week_log2.log 2>&1 &
python -W ignore -u ./batch_test_X.py --Factor_constraint=$7 --indexTarget=$9 --minw=$8 --multitask=$5 --indexName=$6 --indexpull=$4 --DS=$1 --model=$2 --withBJ=$3 --tradeCost 0.0005 --GPU=3 --period=week --cfd=20210101 --cld=20211231 >> week_log3.log 2>&1 &
python -W ignore -u ./batch_test_X.py --Factor_constraint=$7 --indexTarget=$9 --minw=$8 --multitask=$5 --indexName=$6 --indexpull=$4 --DS=$1 --model=$2 --withBJ=$3 --tradeCost 0.0005 --GPU=4 --period=week --cfd=20220101 --cld=20221231 >> week_log4.log 2>&1 &
python -W ignore -u ./batch_test_X.py --Factor_constraint=$7 --indexTarget=$9 --minw=$8 --multitask=$5 --indexName=$6 --indexpull=$4 --DS=$1 --model=$2 --withBJ=$3 --tradeCost 0.0005 --GPU=5 --period=week --cfd=20230101 --cld=20231231 >> week_log5.log 2>&1 &
python -W ignore -u ./batch_test_X.py --Factor_constraint=$7 --indexTarget=$9 --minw=$8 --multitask=$5 --indexName=$6 --indexpull=$4 --DS=$1 --model=$2 --withBJ=$3 --tradeCost 0.0005 --GPU=6 --period=week --cfd=20240101 --cld=20241231 >> week_log6.log 2>&1 &
python -W ignore -u ./batch_test_X.py --Factor_constraint=$7 --indexTarget=$9 --minw=$8 --multitask=$5 --indexName=$6 --indexpull=$4 --DS=$1 --model=$2 --withBJ=$3 --tradeCost 0.0005 --GPU=7 --period=week --cfd=20250101 --cld=20251231 >> week_log7.log 2>&1 &
#python -W ignore -u ./batch_test_L.py --minw=$8 --t=$7 --mask_mf=$5 --indexName=$6 --indexpull=$4 --DS=$1 --model=$2 --withBJ=$3 --GPU=0 --period=week --cfd=20260101 --cld=20260612 >> week_log26.log 2>&1 &


while [ true ];
do
d=`ps -ef|grep batch_test_X.py |wc -l`
if ((d>1)); then
  ((d=d-1))
  ds=`date +"%H:%M:%S"`
  echo "[$ds]还有[$d]进程正在运行中..."
  sleep 15
else
  echo "所有分年回测都已执行完成"
  break
fi
done
echo "生成汇总信息..."
python -W ignore ./batch_test_X.py --SUM=1 --Factor_constraint=$7 --indexTarget=$9 --minw=$8 --multitask=$5 --indexName=$6 --indexpull=$4 --DS=$1 --cfd=20180101 --cld=20260818 --model=$2 --withBJ=$3 --tradeCost 0.0005

# ./kk18_25.sh day196 ./output/Ex0_doublerandom_base_SelfattentAfterAdaln_c196_epoch7.pth 1 0 0 zz1000 1 0.0005 zz1000