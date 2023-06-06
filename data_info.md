# Downloading
The dataset was downloaded from the [LTPP InfoPave](https://infopave.fhwa.dot.gov/) website. 

### Selected Fields:
- Pavement, Structure, and Construction
  - General Section Information
    - GPS Cordinates
    - Improvement History (M&R)
    - Origional Construction Dates
- Climate
  - VWS Data
    - Percipitation
      - Annual
      - Monthly
    - Temperature
      - Annual
      - Monthly
    - Humidity
      - Annual
      - Monthly
  - MERRA Data
    - Termperature, Humidity, Precipitation, Wind, Solar
- Performance
  - Pavement Distress
    - AC
      - Manual Distress
      - HPMS/MEPDG Distress
  - Surface Characteristcs
    - Longitudinal Profile (IRI)
      - Section Level
    - Transverse Profile (Rut)
      - Visit Information
      - Section Level
- Traffic
  - Traffic inputs for pavement analysis
    - Annual Traffic inputs over time
      - Traffic loading characteristics
  - Monitored Traffic Data and Computed Parameters
    - Monitored Traffic Counts
    - Traffic Summary Statistics

### Downloading
These fields were then added to a data bucket and downloaded. The data was downloaded in SQL server backup format. 

# Formatting
The data was then converted from SQL server backup format to CSV format. This was done using the following steps:

*software used: **docker**, **sql-server-management-studio***

First the sql server image was pulled:
```bash
docker pull mcr.microsoft.com/mssql/server:2019-latest
```
Then the server was started with the following command:
```bash
docker run -e 'ACCEPT_EULA=Y' -e 'SA_PASSWORD=<A_PASSWORD_HERE>' -p 1433:1433 --name sql_server -d mcr.microsoft.com/mssql/server:2019-latest
```
Then the .bak files were copied to the container:
```bash
docker cp <FILE_NAME> sql_server:/var/opt/mssql/data
```
Then the .bak files were restored using SSMS.

The restored database was used to load the data into Pandas dataframes. The dataframes were then saved as Parquet Files