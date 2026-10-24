(() => {
  const normalize = value => String(value || "").normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase().replace(/[^a-z0-9]/g, "");
  const drivers = [
  {
    "name": "George Russell",
    "number": 63,
    "code": "RUS",
    "team": "Mercedes-AMG",
    "country": "United Kingdom",
    "flag": "🇬🇧"
  },
  {
    "name": "Kimi Antonelli",
    "number": 12,
    "code": "ANT",
    "team": "Mercedes-AMG",
    "country": "Italy",
    "flag": "🇮🇹"
  },
  {
    "name": "Charles Leclerc",
    "number": 16,
    "code": "LEC",
    "team": "Scuderia Ferrari",
    "country": "Monaco",
    "flag": "🇲🇨"
  },
  {
    "name": "Lewis Hamilton",
    "number": 44,
    "code": "HAM",
    "team": "Scuderia Ferrari",
    "country": "United Kingdom",
    "flag": "🇬🇧"
  },
  {
    "name": "Lando Norris",
    "number": 1,
    "code": "NOR",
    "team": "McLaren",
    "country": "United Kingdom",
    "flag": "🇬🇧"
  },
  {
    "name": "Oscar Piastri",
    "number": 81,
    "code": "PIA",
    "team": "McLaren",
    "country": "Australia",
    "flag": "🇦🇺"
  },
  {
    "name": "Max Verstappen",
    "number": 3,
    "code": "VER",
    "team": "Red Bull Racing",
    "country": "Netherlands",
    "flag": "🇳🇱"
  },
  {
    "name": "Isack Hadjar",
    "number": 6,
    "code": "HAD",
    "team": "Red Bull Racing",
    "country": "France",
    "flag": "🇫🇷"
  },
  {
    "name": "Liam Lawson",
    "number": 30,
    "code": "LAW",
    "team": "Racing Bulls",
    "country": "New Zealand",
    "flag": "🇳🇿"
  },
  {
    "name": "Arvid Lindblad",
    "number": 41,
    "code": "LIN",
    "team": "Racing Bulls",
    "country": "United Kingdom",
    "flag": "🇬🇧"
  },
  {
    "name": "Pierre Gasly",
    "number": 10,
    "code": "GAS",
    "team": "Alpine",
    "country": "France",
    "flag": "🇫🇷"
  },
  {
    "name": "Franco Colapinto",
    "number": 43,
    "code": "COL",
    "team": "Alpine",
    "country": "Argentina",
    "flag": "🇦🇷"
  },
  {
    "name": "Esteban Ocon",
    "number": 31,
    "code": "OCO",
    "team": "Haas",
    "country": "France",
    "flag": "🇫🇷"
  },
  {
    "name": "Oliver Bearman",
    "number": 87,
    "code": "BEA",
    "team": "Haas",
    "country": "United Kingdom",
    "flag": "🇬🇧"
  },
  {
    "name": "Nico Hulkenberg",
    "number": 27,
    "code": "HUL",
    "team": "Audi",
    "country": "Germany",
    "flag": "🇩🇪"
  },
  {
    "name": "Gabriel Bortoleto",
    "number": 5,
    "code": "BOR",
    "team": "Audi",
    "country": "Brazil",
    "flag": "🇧🇷"
  },
  {
    "name": "Carlos Sainz",
    "number": 55,
    "code": "SAI",
    "team": "Williams",
    "country": "Spain",
    "flag": "🇪🇸"
  },
  {
    "name": "Alexander Albon",
    "number": 23,
    "code": "ALB",
    "team": "Williams",
    "country": "Thailand",
    "flag": "🇹🇭"
  },
  {
    "name": "Fernando Alonso",
    "number": 14,
    "code": "ALO",
    "team": "Aston Martin",
    "country": "Spain",
    "flag": "🇪🇸"
  },
  {
    "name": "Lance Stroll",
    "number": 18,
    "code": "STR",
    "team": "Aston Martin",
    "country": "Canada",
    "flag": "🇨🇦"
  },
  {
    "name": "Sergio Perez",
    "number": 11,
    "code": "PER",
    "team": "Cadillac",
    "country": "Mexico",
    "flag": "🇲🇽"
  },
  {
    "name": "Valtteri Bottas",
    "number": 77,
    "code": "BOT",
    "team": "Cadillac",
    "country": "Finland",
    "flag": "🇫🇮"
  }
];
  const teams = [
  {
    "name": "Mercedes-AMG",
    "code": "MER",
    "color": "#27F4D2",
    "aliases": [
      "Mercedes",
      "Mercedes-AMG",
      "Mercedes-AMG Petronas",
      "Mercedes-AMG Petronas Formula One Team"
    ]
  },
  {
    "name": "Scuderia Ferrari",
    "code": "FER",
    "color": "#E8002D",
    "aliases": [
      "Ferrari",
      "Scuderia Ferrari",
      "Scuderia Ferrari HP",
      "Ferrari F1 Team"
    ]
  },
  {
    "name": "McLaren",
    "code": "MCL",
    "color": "#FF8000",
    "aliases": [
      "McLaren",
      "McLaren F1 Team",
      "McLaren Mastercard F1 Team"
    ]
  },
  {
    "name": "Red Bull Racing",
    "code": "RBR",
    "color": "#3671C6",
    "aliases": [
      "Red Bull",
      "Red Bull Racing",
      "Oracle Red Bull Racing"
    ]
  },
  {
    "name": "Racing Bulls",
    "code": "RB",
    "color": "#6692FF",
    "aliases": [
      "RB",
      "VCARB",
      "Racing Bulls",
      "Visa Cash App Racing Bulls",
      "Visa Cash App Racing Bulls Formula One Team"
    ]
  },
  {
    "name": "Alpine",
    "code": "ALP",
    "color": "#00A1E8",
    "aliases": [
      "Alpine",
      "BWT Alpine",
      "BWT Alpine F1 Team"
    ]
  },
  {
    "name": "Aston Martin",
    "code": "AMR",
    "color": "#229971",
    "aliases": [
      "Aston Martin",
      "Aston Martin Aramco",
      "Aston Martin Aramco Formula One Team"
    ]
  },
  {
    "name": "Audi",
    "code": "AUD",
    "color": "#FF2D00",
    "aliases": [
      "Audi",
      "Audi F1 Team",
      "Audi Revolut F1 Team"
    ]
  },
  {
    "name": "Cadillac",
    "code": "CAD",
    "color": "#AAAAAD",
    "aliases": [
      "Cadillac",
      "Cadillac F1 Team",
      "Cadillac Formula 1 Team"
    ]
  },
  {
    "name": "TGR Haas F1 Team",
    "code": "HAS",
    "color": "#DEE1E2",
    "aliases": [
      "Haas",
      "Haas F1 Team",
      "TGR Haas F1 Team",
      "MoneyGram Haas F1 Team"
    ]
  },
  {
    "name": "Williams Racing",
    "code": "WIL",
    "color": "#1868DB",
    "aliases": [
      "Williams",
      "Williams Racing",
      "Atlassian Williams F1 Team"
    ]
  },
  {
    "name": "Sauber",
    "code": "SAU",
    "color": "#52E252",
    "aliases": [
      "Sauber",
      "Kick Sauber",
      "Stake F1 Team",
      "Alfa Romeo"
    ]
  },
  {
    "name": "AlphaTauri",
    "code": "AT",
    "color": "#5E8FAA",
    "aliases": [
      "AlphaTauri",
      "Scuderia AlphaTauri"
    ]
  },
  {
    "name": "Racing Point",
    "code": "RP",
    "color": "#F596C8",
    "aliases": [
      "Racing Point",
      "BWT Racing Point"
    ]
  },
  {
    "name": "Renault",
    "code": "REN",
    "color": "#FFF500",
    "aliases": [
      "Renault",
      "Renault F1 Team"
    ]
  },
  {
    "name": "Toro Rosso",
    "code": "TR",
    "color": "#469BFF",
    "aliases": [
      "Toro Rosso",
      "Scuderia Toro Rosso"
    ]
  }
];
  const teamByName = {};
  teams.forEach(team => team.aliases.forEach(alias => teamByName[normalize(alias)] = team));
  const driverByName = {}, driverByCode = {};
  drivers.forEach(driver => { driverByName[normalize(driver.name)] = driver; driverByCode[driver.code] = driver; });
  const yuki = {"name":"Yuki Tsunoda","number":22,"code":"TSU","team":"Racing Bulls","country":"Japan","flag":"🇯🇵"};
  driverByName[normalize(yuki.name)] = yuki; driverByCode[yuki.code] = yuki;
  driverByName[normalize("Andrea Kimi Antonelli")] = driverByCode.ANT;
  driverByName[normalize("Nico H\u00fclkenberg")] = driverByCode.HUL;
  driverByName[normalize("Nicolas H\u00fclkenberg")] = driverByCode.HUL;
  driverByName[normalize("Nicolas Hulkenberg")] = driverByCode.HUL;
  driverByName[normalize("Sergio P\u00e9rez")] = driverByCode.PER;
  window.F1_DRIVER_INFO = {byName:driverByName,byCode:driverByCode};
  window.F1_TEAM_INFO = teamByName;
  const grid = document.getElementById("driver-grid");
  if (!grid) return;
  drivers.forEach(driver => {
    const team = teamByName[normalize(driver.team)];
    const card = document.createElement("article");
    card.className = "driver-card";
    card.setAttribute("aria-label",driver.name+", number "+driver.number+", "+team.name+", "+driver.country);
    card.style.setProperty("--team-colour",team.color);
    const identity = document.createElement("div");
    identity.className = "driver-identity";
    identity.style.cssText = "height:190px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:3px;position:relative;border-bottom:1px solid var(--border);background:radial-gradient(ellipse at 50% 100%,color-mix(in srgb,"+team.color+" 28%,transparent),transparent 72%),#101113";
    const number = document.createElement("strong");
    number.className = "driver-number"; number.textContent = "#"+driver.number;
    number.style.cssText = "position:static;border:0;background:transparent;font-size:clamp(46px,6vw,72px);line-height:.95;padding:0;color:"+team.color;
    const code = document.createElement("span");
    code.className = "driver-code"; code.textContent = driver.code;
    code.style.cssText = "font:700 13px/1 monospace;letter-spacing:.2em;color:#f3f2f1";
    const flag = document.createElement("span");
    flag.className = "driver-flag-emoji"; flag.textContent = driver.flag; flag.setAttribute("aria-hidden","true");
    flag.style.cssText = "font:28px/1 system-ui,sans-serif;margin-top:8px";
    identity.append(number,code,flag);
    const details = document.createElement("div"); details.className = "driver-details";
    const name = document.createElement("h3"); name.className = "driver-name"; name.textContent = driver.name;
    const teamName = document.createElement("p"); teamName.className = "driver-team"; teamName.textContent = team.name;
    const country = document.createElement("p"); country.className = "driver-country"; country.textContent = driver.flag+" "+driver.country;
    details.append(name,teamName,country); card.append(identity,details); grid.append(card);
  });
})();
