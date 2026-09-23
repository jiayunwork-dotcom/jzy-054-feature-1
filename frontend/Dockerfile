# ---- build stage: Node.js 20 ----
FROM node:20-bookworm-slim AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# ---- serve stage ----
FROM nginx:1.27-alpine

# Static assets + SPA entry
COPY --from=build /app/dist /usr/share/nginx/html
# Reverse-proxy /api/* to the backend container
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
