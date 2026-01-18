from dataclasses import dataclass

@dataclass(frozen=True)
class LeagueConfig:
    code: str
    fd_code: str
    seasons: list[str]

LEAGUES = [
    LeagueConfig(code="EPL", fd_code="E0",  seasons=["2526", "2425", "2324", "2223", "2122"]),
    LeagueConfig(code="SA",  fd_code="I1",  seasons=["2526", "2425", "2324", "2223", "2122"]),
    LeagueConfig(code="LL",  fd_code="SP1", seasons=["2526", "2425", "2324", "2223", "2122"]),  # LaLiga
]

FOOTBALL_DATA_BASE = "https://www.football-data.co.uk/mmz4281"
