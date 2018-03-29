#!/bin/sh

sed -i "s#\%ADMIN_PASSWORD\%#$ADMIN_PASSWORD#g" /jibiki/papillon.properties

sed -i "s#\%SPECIALIST_PASSWORD\%#$SPECIALIST_PASSWORD#g" /jibiki/papillon.properties

sed -i "s#\%DATABASE_HOST\%#$DATABASE_HOST#g" /jibiki/papillon.properties

sed -i "s#\%DATABASE_NAME\%#$DATABASE_NAME#g" /jibiki/papillon.properties

sed -i "s#\%DATABASE_USER\%#$DATABASE_USER#g" /jibiki/papillon.properties

sed -i "s#\%DATABASE_PASSWORD\%#$DATABASE_PASSWORD#g" /jibiki/papillon.properties


