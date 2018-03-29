#!/bin/bash
set -e

WORKDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" -c "CREATE USER $DATABASE_USER WITH PASSWORD '$DATABASE_PASSWORD' CREATEDB;"

psql -v ON_ERROR_STOP=1 --username "$DATABASE_USER" "$POSTGRES_USER" -c "CREATE DATABASE $DATABASE_NAME WITH ENCODING 'UTF-8';"

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" "$DATABASE_NAME" -f ${WORKDIR}/sql/multilingual-sort.sql

psql -v ON_ERROR_STOP=1 --username "$DATABASE_USER" "$DATABASE_NAME" -f ${WORKDIR}/sql/SQLcreate.sql