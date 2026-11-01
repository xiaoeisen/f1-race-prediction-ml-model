(() => {
  const drivers = [
    { name: "George Russell", number: 63, team: "Mercedes", country: "United Kingdom" },
    { name: "Kimi Antonelli", number: 12, team: "Mercedes", country: "Italy" },
    { name: "Charles Leclerc", number: 16, team: "Ferrari", country: "Monaco" },
    { name: "Lewis Hamilton", number: 44, team: "Ferrari", country: "United Kingdom" },
    { name: "Lando Norris", number: 1, team: "McLaren", country: "United Kingdom" },
    { name: "Oscar Piastri", number: 81, team: "McLaren", country: "Australia" },
    { name: "Max Verstappen", number: 3, team: "Red Bull Racing", country: "Netherlands" },
    { name: "Isack Hadjar", number: 6, team: "Red Bull Racing", country: "France" },
    { name: "Liam Lawson", number: 30, team: "Racing Bulls", country: "New Zealand" },
    { name: "Arvid Lindblad", number: 41, team: "Racing Bulls", country: "United Kingdom" },
    { name: "Pierre Gasly", number: 10, team: "Alpine", country: "France" },
    { name: "Franco Colapinto", number: 43, team: "Alpine", country: "Argentina" },
    { name: "Esteban Ocon", number: 31, team: "Haas F1 Team", country: "France" },
    { name: "Oliver Bearman", number: 87, team: "Haas F1 Team", country: "United Kingdom" },
    { name: "Nico Hulkenberg", number: 27, team: "Audi", country: "Germany" },
    { name: "Gabriel Bortoleto", number: 5, team: "Audi", country: "Brazil" },
    { name: "Carlos Sainz", number: 55, team: "Williams", country: "Spain" },
    { name: "Alexander Albon", number: 23, team: "Williams", country: "Thailand" },
    { name: "Fernando Alonso", number: 14, team: "Aston Martin", country: "Spain" },
    { name: "Lance Stroll", number: 18, team: "Aston Martin", country: "Canada" },
    { name: "Sergio Perez", number: 11, team: "Cadillac", country: "Mexico" },
    { name: "Valtteri Bottas", number: 77, team: "Cadillac", country: "Finland" },
  ];
  const countryFlagSvg = {
    GB: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 40"><rect width="60" height="40" fill="#012169"/><path d="M0 0 60 40M60 0 0 40" stroke="#fff" stroke-width="10"/><path d="M0 0 60 40M60 0 0 40" stroke="#c8102e" stroke-width="4"/><path d="M30 0v40M0 20h60" stroke="#fff" stroke-width="14"/><path d="M30 0v40M0 20h60" stroke="#c8102e" stroke-width="8"/></svg>`,
    IT: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 40"><path fill="#009246" d="M0 0h20v40H0z"/><path fill="#fff" d="M20 0h20v40H20z"/><path fill="#ce2b37" d="M40 0h20v40H40z"/></svg>`,
    MC: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 40"><path fill="#ce1126" d="M0 0h60v20H0z"/><path fill="#fff" d="M0 20h60v20H0z"/></svg>`,
    AU: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 40"><rect width="60" height="40" fill="#00008b"/><svg x="0" y="0" width="28" height="19" viewBox="0 0 60 40"><rect width="60" height="40" fill="#012169"/><path d="M0 0 60 40M60 0 0 40" stroke="#fff" stroke-width="10"/><path d="M0 0 60 40M60 0 0 40" stroke="#c8102e" stroke-width="4"/><path d="M30 0v40M0 20h60" stroke="#fff" stroke-width="14"/><path d="M30 0v40M0 20h60" stroke="#c8102e" stroke-width="8"/></svg><g fill="#fff"><circle cx="42" cy="9" r="2"/><circle cx="50" cy="19" r="2"/><circle cx="38" cy="25" r="2"/><circle cx="51" cy="32" r="2"/><circle cx="27" cy="33" r="2"/></g></svg>`,
    NL: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 40"><path fill="#ae1c28" d="M0 0h60v13.33H0z"/><path fill="#fff" d="M0 13.33h60v13.34H0z"/><path fill="#21468b" d="M0 26.67h60V40H0z"/></svg>`,
    FR: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 40"><path fill="#0055a4" d="M0 0h20v40H0z"/><path fill="#fff" d="M20 0h20v40H20z"/><path fill="#ef4135" d="M40 0h20v40H40z"/></svg>`,
    NZ: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 40"><rect width="60" height="40" fill="#00247d"/><svg x="0" y="0" width="28" height="19" viewBox="0 0 60 40"><rect width="60" height="40" fill="#012169"/><path d="M0 0 60 40M60 0 0 40" stroke="#fff" stroke-width="10"/><path d="M0 0 60 40M60 0 0 40" stroke="#c8102e" stroke-width="4"/><path d="M30 0v40M0 20h60" stroke="#fff" stroke-width="14"/><path d="M30 0v40M0 20h60" stroke="#c8102e" stroke-width="8"/></svg><g fill="#c8102e" stroke="#fff" stroke-width="1.5"><path d="m43 6 1.5 4h4l-3.2 2.4 1.2 4-3.5-2.5-3.5 2.5 1.2-4-3.2-2.4h4z"/><path d="m52 17 1.5 4h4l-3.2 2.4 1.2 4-3.5-2.5-3.5 2.5 1.2-4-3.2-2.4h4z"/><path d="m42 27 1.5 4h4l-3.2 2.4 1.2 4-3.5-2.5-3.5 2.5 1.2-4-3.2-2.4h4z"/></g></svg>`,
    DE: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 40"><path fill="#000" d="M0 0h60v13.33H0z"/><path fill="#dd0000" d="M0 13.33h60v13.34H0z"/><path fill="#ffce00" d="M0 26.67h60V40H0z"/></svg>`,
    BR: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 40"><rect width="60" height="40" fill="#009b3a"/><path fill="#ffdf00" d="m30 4 23 16-23 16L7 20z"/><circle cx="30" cy="20" r="9" fill="#002776"/><path d="M22 18q8-4 16 1" fill="none" stroke="#fff" stroke-width="1.5"/></svg>`,
    ES: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 40"><path fill="#aa151b" d="M0 0h60v10H0zM0 30h60v10H0z"/><path fill="#f1bf00" d="M0 10h60v20H0z"/></svg>`,
    TH: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 40"><path fill="#a51931" d="M0 0h60v6.67H0zM0 33.33h60V40H0z"/><path fill="#f4f5f8" d="M0 6.67h60v6.66H0zM0 26.67h60v6.66H0z"/><path fill="#2d2a4a" d="M0 13.33h60v13.34H0z"/></svg>`,
    MX: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 40"><path fill="#006847" d="M0 0h20v40H0z"/><path fill="#fff" d="M20 0h20v40H20z"/><path fill="#ce1126" d="M40 0h20v40H40z"/><circle cx="30" cy="20" r="3" fill="#9b7b35"/></svg>`,
    FI: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 40"><rect width="60" height="40" fill="#fff"/><path fill="#003580" d="M16 0h8v40h-8zM0 16h60v8H0z"/></svg>`,
    AR: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 40"><path fill="#74acdf" d="M0 0h60v13.33H0zM0 26.67h60V40H0z"/><path fill="#fff" d="M0 13.33h60v13.34H0z"/><circle cx="30" cy="20" r="3" fill="#f6b40e"/></svg>`,
    CA: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 40"><path fill="#d80621" d="M0 0h15v40H0zM45 0h15v40H45z"/><path fill="#fff" d="M15 0h30v40H15z"/><path fill="#d80621" d="m30 7 3 7 5-2-2 6 5 3-6 2 1 8-6-4-6 4 1-8-6-2 5-3-2-6 5 2z"/></svg>`
  };
  const countryFlagDataUrl = (code) => `data:image/svg+xml;charset=utf-8,${encodeURIComponent(countryFlagSvg[code])}`;  const grid = document.getElementById("driver-grid");
  const countryFlags = { "United Kingdom": "GB", Italy: "IT", Monaco: "MC", Australia: "AU", Netherlands: "NL", France: "FR", "New Zealand": "NZ", Germany: "DE", Brazil: "BR", Spain: "ES", Thailand: "TH", Mexico: "MX", Finland: "FI", Argentina: "AR", Canada: "CA" };
  const driverCountryByCode = { RUS: "United Kingdom", ANT: "Italy", LEC: "Monaco", HAM: "United Kingdom", NOR: "United Kingdom", PIA: "Australia", VER: "Netherlands", HAD: "France", LAW: "New Zealand", LIN: "United Kingdom", GAS: "France", COL: "Argentina", OCO: "France", BEA: "United Kingdom", HUL: "Germany", BOR: "Brazil", SAI: "Spain", ALB: "Thailand", ALO: "Spain", STR: "Canada", PER: "Mexico", BOT: "Finland" };
  window.F1_DRIVER_FLAGS_BY_CODE = Object.fromEntries(Object.entries(driverCountryByCode).map(([code, country]) => [code, countryFlagDataUrl(countryFlags[country])]));
  if (!grid) return;
  drivers.forEach((driver) => {
    const card = document.createElement("article");
    card.className = "driver-card";
    card.setAttribute("aria-label", driver.name + ", car number " + driver.number + ", " + driver.team + ", " + driver.country);

    const portrait = document.createElement("div");
    portrait.className = "driver-portrait";
    portrait.style.setProperty("--team-colour", driver.team === "Ferrari" ? "#e8002d" :
      driver.team === "Mercedes" ? "#27b5a0" : driver.team === "McLaren" ? "#ff8000" :
      driver.team.indexOf("Red Bull") === 0 ? "#3671c6" : "#ff6848");
    portrait.append(Object.assign(document.createElement("span"), {
      className: "driver-number",
      textContent: "#" + driver.number,
    }));

    const country = document.createElement("p");
    country.className = "driver-country";
    const flagCode = countryFlags[driver.country];
    if (flagCode) {
      const flag = document.createElement("img");
      flag.className = "driver-flag";
      flag.alt = "";
      flag.setAttribute("aria-hidden", "true");
      flag.loading = "lazy";
      flag.decoding = "async";
      flag.referrerPolicy = "no-referrer";
      flag.src = countryFlagDataUrl(flagCode);
      country.append(flag);
    }
    country.append(document.createTextNode(driver.country));

    const details = document.createElement("div");
    details.className = "driver-details";
    details.append(
      Object.assign(document.createElement("h3"), { className: "driver-name", textContent: driver.name }),
      Object.assign(document.createElement("p"), { className: "driver-team", textContent: driver.team }),
      country
    );
    card.append(portrait, details);
    grid.append(card);
  });
})();

