#!/bin/sh

STARDATE='20160401'
ENDDATE='20170131'
# 環境変数APIKEYにフレームワークスのAPIキーを保持させておく
# APIKEY=<YOUR_API_KEY>

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
