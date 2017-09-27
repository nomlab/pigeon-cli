#!/bin/sh

STARDATE=$1 # 20160401
ENDDATE=$2  # 20170131
APIKEY=$3   # <YOUR_APIKEY>

if [ ! -e ./json ] ; then
    mkdir ./json
fi

CURRENTDATE=$STARDATE
while [ 1 ] ; do
    curl -sL https://api.frameworxopendata.jp/api/v3/files/hems/${CURRENTDATE}.json?acl:consumerKey=$APIKEY -o ./json/${CURRENTDATE}.json
    echo 'Done' ${CURRENTDATE}.json
    if [ $CURRENTDATE = $ENDDATE ] ; then
        break
    fi
    CURRENTDATE=`date -j -v+1d -f "%Y%m%d" $CURRENTDATE +%Y%m%d`
done
