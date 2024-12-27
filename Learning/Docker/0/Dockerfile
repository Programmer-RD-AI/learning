FROM node:12

WORKDIR /app

# Local Location, Destination Location
COPY package*.json ./

RUN npm install

COPY . .

ENV PORT=8080

EXPOSE 8080

# Exec Form 
CMD ["npm", "start"]
