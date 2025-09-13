FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
ARG PROJECT
WORKDIR /repo
COPY proto ./proto
COPY src/dotnet ./src/dotnet
RUN dotnet publish "src/dotnet/${PROJECT}/${PROJECT}.csproj" -c Release -o /out --nologo -v q

FROM mcr.microsoft.com/dotnet/aspnet:10.0
ARG PROJECT
RUN apt-get update && apt-get install -y --no-install-recommends curl tzdata && rm -rf /var/lib/apt/lists/*
COPY --from=build /out /app
WORKDIR /app
RUN ln -s "/app/${PROJECT}.dll" /app/service.dll
ENTRYPOINT ["dotnet", "/app/service.dll"]
