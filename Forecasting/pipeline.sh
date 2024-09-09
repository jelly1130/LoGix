#!/bin/bash


if [ -n "$1" ]; then
    # $1 is not null, do something
    m=$1
    sh scripts/$m.sh > ori_result.$m.txt
    bash grep_result.sh $m > msg.md
else
    # $1 is null or empty, do something else
    echo 'please specify message m'
fi