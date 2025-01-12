SELECT
    TOP (1000) [optname],
    [value],
    [major_version],
    [minor_version],
    [revision],
    [install_failures]
FROM
    [master].[dbo].[MSreplication_options]
