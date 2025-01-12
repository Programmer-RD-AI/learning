FROM node:19.4-bullseye as build

WORKDIR /usr/src/app

COPY package*.json ./

RUN --mount=type=cache,target=/usr/src/app/.npm \
    npm set cache /usr/src/app/.npm \
    npm ci

COPY . .

RUN npm run build

FROM nginxinc/nginx-unprivileged:1.22-alpine

COPY --from=build /usr/src/app/build /usr/share/nginx/html

COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 8080

CMD ["npm", "run", "dev"]

