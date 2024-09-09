#!/bin/bash

m=$1
fn=ori_result.$m.txt
mse_avg=`cat $fn|grep result|grep MSE|sed 's/,//g'|awk '{a+=$4;b+=1}END{print a/b}'`
mae_avg=`cat $fn|grep result|grep MAE|sed 's/,//g'|awk '{a+=$4;b+=1}END{print a/b}'`
if [[ $mse_avg =~ ^[0-9.]+$ ]];then
    echo "# JA800($m): Mixer forcast training successfully"
else
    echo "# JA800($m): Mixer forcast training Failed"
fi

echo "**MSE**: "$mse_avg "**MAE**:" $mae_avg
echo "===================================================================="
cat $fn|grep result|awk '{if(NR % 2){print int(NR/2)+1"."} print $0}'