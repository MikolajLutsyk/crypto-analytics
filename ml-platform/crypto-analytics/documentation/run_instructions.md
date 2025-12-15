
for backend:

uvicorn main:app --reload --host 0.0.0.0 --port 8000

for frontend:

npm start

to run database contatiner:

docker-compose -up

to connect to the db:

psql -h localhost -U postgres -d crypto
