from frozendict import frozendict

# Maps place of birth to coarse categories for state
# or global region (i.e. Africa, Middle East, South America).
POBP_MAPPING = {
    001.: "AL",  # .Alabama/AL
    002.: "AK",  # .Alaska/AK
    004.: "AZ",  # .Arizona/AZ
    005.: "AR",  # .Arkansas/AR
    006.: "CA",  # .California/CA
    008.: "CO",  # .Colorado/CO
    009.: "CT",  # .Connecticut/CT
    010.: "DE",  # .Delaware/DE
    011.: "DC",  # .District of Columbia/DC
    012.: "FL",  # .Florida/FL
    013.: "GA",  # .Georgia/GA
    015.: "HI",  # .Hawaii/HI
    016.: "ID",  # .Idaho/ID
    017.: "IL",  # .Illinois/IL
    018.: "IN",  # .Indiana/IN
    019.: "IA",  # .Iowa/IA
    020.: "KS",  # .Kansas/KS
    021.: "KY",  # .Kentucky/KY
    022.: "LA",  # .Louisiana/LA
    023.: "ME",  # .Maine/ME
    024.: "MD",  # .Maryland/MD
    025.: "MA",  # .Massachusetts/MA
    026.: "MI",  # .Michigan/MI
    027.: "MN",  # .Minnesota/MN
    028.: "MS",  # .Mississippi/MS
    029.: "MO",  # .Missouri/MO
    030.: "MT",  # .Montana/MT
    031.: "NE",  # .Nebraska/NE
    032.: "NV",  # .Nevada/NV
    033.: "NH",  # .New Hampshire/NH
    034.: "NJ",  # .New Jersey/NJ
    035.: "NM",  # .New Mexico/NM
    036.: "NY",  # .New York/NY
    037.: "NC",  # .North Carolina/NC
    038.: "ND",  # .North Dakota/ND
    039.: "OH",  # .Ohio/OH
    040.: "OK",  # .Oklahoma/OK
    041.: "OR",  # .Oregon/OR
    042.: "PA",  # .Pennsylvania/PA
    044.: "RI",  # .Rhode Island/RI
    045.: "SC",  # .South Carolina/SC
    046.: "SD",  # .South Dakota/SD
    047.: "TN",  # .Tennessee/TN
    048.: "TX",  # .Texas/TX
    049.: "UT",  # .Utah/UT
    050.: "VT",  # .Vermont/VT
    051.: "VA",  # .Virginia/VA
    053.: "WA",  # .Washington/WA
    054.: "WV",  # .West Virginia/WV
    055.: "WI",  # .Wisconsin/WI
    056.: "WY",  # .Wyoming/WY
    060.: "USTERR",  # .American Samoa
    066.: "USTERR",  # .Guam
    069.: "USTERR",  # .Commonwealth of the Northern Mariana Islands
    072.: "PR",  # .Puerto Rico
    078.: "USTERR",  # .US Virgin Islands
    100.: "EUROPE",  # .Albania
    102.: "EUROPE",  # .Austria
    103.: "EUROPE",  # .Belgium
    104.: "EUROPE",  # .Bulgaria
    105.: "EUROPE",  # .Czechoslovakia
    106.: "EUROPE",  # .Denmark
    108.: "EUROPE",  # .Finland
    109.: "EUROPE",  # .France
    110.: "EUROPE",  # .Germany
    116.: "EUROPE",  # .Greece
    117.: "EUROPE",  # .Hungary
    118.: "EUROPE",  # .Iceland
    119.: "EUROPE",  # .Ireland
    120.: "EUROPE",  # .Italy
    126.: "EUROPE",  # .Netherlands
    127.: "EUROPE",  # .Norway
    128.: "EUROPE",  # .Poland
    129.: "EUROPE",  # .Portugal
    130.: "EUROPE",  # .Azores Islands
    132.: "EUROPE",  # .Romania
    134.: "EUROPE",  # .Spain
    136.: "EUROPE",  # .Sweden
    137.: "EUROPE",  # .Switzerland
    138.: "EUROPE",  # .United Kingdom, Not Specified
    139.: "EUROPE",  # .England
    140.: "EUROPE",  # .Scotland
    142.: "EUROPE",  # .Northern Ireland (201.: "", #7 or later)
    147.: "EUROPE",  # .Yugoslavia
    148.: "EUROPE",  # .Czech Republic
    149.: "EUROPE",  # .Slovakia
    150.: "EUROPE",  # .Bosnia and Herzegovina
    151.: "EUROPE",  # .Croatia
    152.: "EUROPE",  # .Macedonia
    154.: "EUROPE",  # .Serbia
    156.: "EUROPE",  # .Latvia
    157.: "EUROPE",  # .Lithuania
    158.: "EUROPE",  # .Armenia
    159.: "EUROPE",  # .Azerbaijan
    160.: "EUROPE",  # .Belarus
    161.: "EUROPE",  # .Georgia
    162.: "EUROPE",  # .Moldova
    163.: "ASIA",  # .Russia
    164.: "ASIA",  # .Ukraine
    165.: "ASIA",  # .USSR
    166.: "EUROPE",  # .Europe (201.: "", #7 or later)
    167.: "EUROPE",  # .Kosovo (201.: "", #7 or later)
    168.: "EUROPE",  # .Montenegro
    169.: "EUROPE",  # .Other Europe, Not Specified
    200.: "MIDDLEEAST",  # .Afghanistan
    202.: "ASIA",  # .Bangladesh
    203.: "ASIA",  # .Bhutan
    205.: "ASIA",  # .Myanmar
    206.: "ASIA",  # .Cambodia
    207.: "ASIA",  # .China
    208.: "EUROPE",  # .Cyprus (201.: "", #6 or earlier)
    209.: "ASIA",  # .Hong Kong
    210.: "ASIA",  # .India
    211.: "ASIA",  # .Indonesia
    212.: "MIDDLEEAST",  # .Iran
    213.: "MIDDLEEAST",  # .Iraq
    214.: "MIDDLEEAST",  # .Israel
    215.: "ASIA",  # .Japan
    216.: "MIDDLEEAST",  # .Jordan
    217.: "ASIA",  # .Korea
    218.: "MIDDLEEAST",  # .Kazakhstan
    219.: "MIDDLEEAST",  # .Kyrgyzstan (201.: "", #7 or later)
    222.: "MIDDLEEAST",  # .Kuwait
    223.: "ASIA",  # .Laos
    224.: "MIDDLEEAST",  # .Lebanon
    226.: "ASIA",  # .Malaysia
    228.: "ASIA",  # .Mongolia (201.: "", #7 or later)
    229.: "ASIA",  # .Nepal
    231.: "ASIA",  # .Pakistan
    233.: "ASIA",  # .Philippines
    235.: "ASIA",  # .Saudi Arabia
    236.: "ASIA",  # .Singapore
    238.: "ASIA",  # .Sri Lanka
    239.: "ASIA",  # .Syria
    240.: "ASIA",  # .Taiwan
    242.: "ASIA",  # .Thailand
    243.: "EUROPE",  # .Turkey
    245.: "MIDDLEEAST",  # .United Arab Emirates
    246.: "MIDDLEEAST",  # .Uzbekistan
    247.: "ASIA",  # .Vietnam
    248.: "MIDDLEEAST",  # .Yemen
    249.: "ASIA",  # .Asia
    253.: "ASIA",  # .South Central Asia, Not Specified
    254.: "ASIA",  # .Other Asia, Not Specified
    300.: "CARIBBEAN",  # .Bermuda
    301.: "CANADA",  # .Canada
    303.: "MEXICO",  # .Mexico
    310.: "CENTRALAMERICA",  # .Belize
    311.: "CENTRALAMERICA",  # .Costa Rica
    312.: "CENTRALAMERICA",  # .El Salvador
    313.: "CENTRALAMERICA",  # .Guatemala
    314.: "CENTRALAMERICA",  # .Honduras
    315.: "CENTRALAMERICA",  # .Nicaragua
    316.: "CENTRALAMERICA",  # .Panama
    321.: "CARIBBEAN",  # .Antigua and Barbuda
    323.: "CARIBBEAN",  # .Bahamas
    324.: "CARIBBEAN",  # .Barbados
    327.: "CARIBBEAN",  # .Cuba
    328.: "CARIBBEAN",  # .Dominica
    329.: "CARIBBEAN",  # .Dominican Republic
    330.: "CARIBBEAN",  # .Grenada
    332.: "CARIBBEAN",  # .Haiti
    333.: "CARIBBEAN",  # .Jamaica
    338.: "CARIBBEAN",  # .St. Kitts-Nevis (201.: "", #7 or later)
    339.: "CARIBBEAN",  # .St. Lucia
    340.: "CARIBBEAN",  # .St. Vincent and the Grenadines
    341.: "CARIBBEAN",  # .Trinidad and Tobago
    343.: "CARIBBEAN",  # .West Indies
    344.: "CARIBBEAN",  # .Caribbean, Not Specified
    360.: "SOUTHAMERICA",  # .Argentina
    361.: "SOUTHAMERICA",  # .Bolivia
    362.: "SOUTHAMERICA",  # .Brazil
    363.: "SOUTHAMERICA",  # .Chile
    364.: "SOUTHAMERICA",  # .Colombia
    365.: "SOUTHAMERICA",  # .Ecuador
    368.: "SOUTHAMERICA",  # .Guyana
    369.: "SOUTHAMERICA",  # .Paraguay
    370.: "SOUTHAMERICA",  # .Peru
    372.: "SOUTHAMERICA",  # .Uruguay
    373.: "SOUTHAMERICA",  # .Venezuela
    374.: "SOUTHAMERICA",  # .South America
    399.: "SOUTHAMERICA",  # .Americas, Not Specified
    400.: "AFRICA",  # .Algeria
    407.: "AFRICA",  # .Cameroon
    408.: "AFRICA",  # .Cabo Verde
    412.: "AFRICA",  # .Congo
    414.: "AFRICA",  # .Egypt
    416.: "AFRICA",  # .Ethiopia
    417.: "AFRICA",  # .Eritrea
    420.: "AFRICA",  # .Gambia
    421.: "AFRICA",  # .Ghana
    423.: "AFRICA",  # .Guinea
    425.: "AFRICA",  # .Ivory Coast (201.: "", #7 or later)
    427.: "AFRICA",  # .Kenya
    429.: "AFRICA",  # .Liberia
    430.: "AFRICA",  # .Libya
    436.: "AFRICA",  # .Morocco
    440.: "AFRICA",  # .Nigeria
    442.: "AFRICA",  # .Rwanda (201.: "", #7 or later)
    444.: "AFRICA",  # .Senegal
    447.: "AFRICA",  # .Sierra Leone
    448.: "AFRICA",  # .Somalia
    449.: "AFRICA",  # .South Africa
    451.: "AFRICA",  # .Sudan
    453.: "AFRICA",  # .Tanzania
    454.: "AFRICA",  # .Togo
    456.: "AFRICA",  # .Tunisia (201.: "", #7 or later)
    457.: "AFRICA",  # .Uganda
    459.: "AFRICA",  # .Democratic Republic of Congo (Zaire)
    460.: "AFRICA",  # .Zambia
    461.: "AFRICA",  # .Zimbabwe
    462.: "AFRICA",  # .Africa
    463.: "AFRICA",  # .South Sudan (201.: "", #7 or later)
    464.: "AFRICA",  # .Northern Africa, Not Specified
    467.: "AFRICA",  # .Western Africa, Not Specified
    468.: "AFRICA",  # .Other Africa, Not Specified
    469.: "AFRICA",  # .Eastern Africa, Not Specified
    501.: "OCEANIA",  # .Australia
    508.: "OCEANIA",  # .Fiji
    511.: "OCEANIA",  # .Marshall Islands
    512.: "OCEANIA",  # .Micronesia
    515.: "OCEANIA",  # .New Zealand
    523.: "OCEANIA",  # .Tonga
    527.: "OCEANIA",  # .Samoa
    554.: "OCEANIA",  # .Other US Island Areas, Oceania, Not Specified, or at Sea
}

OCCP_MAPPING_FINE = {
    "bbbb": "", # .N/A (less than 16 years old/NILF who last worked more than .5 years ago or never worked)
    0010.: "00", # # .MGR-Chief Executives And Legislators
    0020.: "00", # # .MGR-General And Operations Managers
    0040.: "00", # # .MGR-Advertising And Promotions Managers
    0051.: "00", # # .MGR-Marketing Managers
    0052.: "00", # # .MGR-Sales Managers
    0060.: "00", # # .MGR-Public Relations And Fundraising Managers
    0101.: "01", # # .MGR-Administrative Services Managers
    0102.: "01", # # .MGR-Facilities Managers
    0110.: "01", # # .MGR-Computer And Information Systems Managers
    0120.: "01", # # .MGR-Financial Managers
    0135.: "01", # # .MGR-Compensation And Benefits Managers
    0136.: "01", # # .MGR-Human Resources Managers
    0137.: "01", # # .MGR-Training And Development Managers
    0140.: "01", # # .MGR-Industrial Production Managers
    0150.: "01", # # .MGR-Purchasing Managers
    0160.: "01", # # .MGR-Transportation, Storage, And Distribution Managers
    0205.: "02", # # .MGR-Farmers, Ranchers, And Other Agricultural Managers
    0220.: "02", # # .MGR-Construction Managers
    0230.: "02", # # .MGR-Education And Childcare Administrators
    0300.: "03", # # .MGR-Architectural And Engineering Managers
    0310.: "03", # # .MGR-Food Service Managers
    0335.: "03", # # .MGR-Entertainment and Recreation Managers
    0340.: "03", # # .MGR-Lodging Managers
    0350.: "03", # # .MGR-Medical And Health Services Managers
    0360.: "03", # # .MGR-Natural Sciences Managers
    0410.: "04", # # .MGR-Property, Real Estate, And Community Association .Managers
    0420.: "04", # # .MGR-Social And Community Service Managers
    0425.: "04", # # .MGR-Emergency Management Directors
    0440.: "04", # # .MGR-Other Managers
    0500.: "05", # # .BUS-Agents And Business Managers Of Artists, Performers, .And Athletes
    0510.: "05", # # .BUS-Buyers And Purchasing Agents, Farm Products
    0520.: "05", # # .BUS-Wholesale And Retail Buyers, Except Farm Products
    0530.: "05", # # .BUS-Purchasing Agents, Except Wholesale, Retail, And Farm .Products
    0540.: "05", # # .BUS-Claims Adjusters, Appraisers, Examiners, And .Investigators
    0565.: "05", # # .BUS-Compliance Officers
    0600.: "06", # # .BUS-Cost Estimators
    0630.: "06", # # .BUS-Human Resources Workers
    0640.: "06", # # .BUS-Compensation, Benefits, And Job Analysis Specialists
    0650.: "06", # # .BUS-Training And Development Specialists
    0700.: "07", # # .BUS-Logisticians
    0705.: "07", # # .BUS-Project Management Specialists
    0710.: "07", # # .BUS-Management Analysts
    0725.: "07", # # .BUS-Meeting, Convention, And Event Planners
    0726.: "07", # # .BUS-Fundraisers
    0735.: "07", # # .BUS-Market Research Analysts And Marketing Specialists
    0750.: "07", # # .BUS-Business Operations Specialists, All Other
    0800.: "08", # # .FIN-Accountants And Auditors
    0810.: "08", # # .FIN-Property Appraisers and Assessors
    0820.: "08", # # .FIN-Budget Analysts
    0830.: "08", # # .FIN-Credit Analysts
    0845.: "08", # # .FIN-Financial And Investment Analysts
    0850.: "08", # # .FIN-Personal Financial Advisors
    0860.: "08", # # .FIN-Insurance Underwriters
    0900.: "09", # # .FIN-Financial Examiners
    0910.: "09", # # .FIN-Credit Counselors And Loan Officers
    0930.: "09", # # .FIN-Tax Examiners And Collectors, And Revenue Agents
    0940.: "09", # # .FIN-Tax Preparers
    0960.: "09", # # .FIN-Other Financial Specialists
    1005.: "10", # # .CMM-Computer And Information Research Scientists
    1006.: "10", # # .CMM-Computer Systems Analysts
    1007.: "10", # # .CMM-Information Security Analysts
    1010.: "10", # # .CMM-Computer Programmers
    1021.: "10", # # .CMM-Software Developers
    1022.: "10", # # .CMM-Software Quality Assurance Analysts and Testers
    1031.: "10", # # .CMM-Web Developers
    1032.: "10", # # .CMM-Web And Digital Interface Designers
    1050.: "10", # # .CMM-Computer Support Specialists
    1065.: "10", # # .CMM-Database Administrators and Architects
    1105.: "11", # # .CMM-Network And Computer Systems Administrators
    1106.: "11", # # .CMM-Computer Network Architects
    1108.: "11", # # .CMM-Computer Occupations, All Other
    1200.: "12", # # .CMM-Actuaries
    1220.: "12", # # .CMM-Operations Research Analysts
    1240.: "12", # # .CMM-Other Mathematical Science Occupations
    1305.: "13", # # .ENG-Architects, Except Landscape And Naval
    1306.: "13", # # .ENG-Landscape Architects
    1310.: "13", # # .ENG-Surveyors, Cartographers, And Photogrammetrists
    1320.: "13", # # .ENG-Aerospace Engineers
    1340.: "13", # # .ENG-Biomedical And Agricultural Engineers
    1350.: "13", # # .ENG-Chemical Engineers
    1360.: "13", # # .ENG-Civil Engineers
    1400.: "14", # # .ENG-Computer Hardware Engineers
    1410.: "14", # # .ENG-Electrical And Electronics Engineers
    1420.: "14", # # .ENG-Environmental Engineers
    1430.: "14", # # .ENG-Industrial Engineers, Including Health And Safety
    1440.: "14", # # .ENG-Marine Engineers And Naval Architects
    1450.: "14", # # .ENG-Materials Engineers
    1460.: "14", # # .ENG-Mechanical Engineers
    1520.: "15", # # .ENG-Petroleum, Mining And Geological Engineers, Including .Mining Safety Engineers
    1530.: "15", # # .ENG-Other Engineers
    1541.: "15", # # .ENG-Architectural And Civil Drafters
    1545.: "15", # # .ENG-Other Drafters
    1551.: "15", # # .ENG-Electrical And Electronic Engineering Technologists and .Technicians
    1555.: "15", # # .Other Engineering Technologists And Technicians, Except .Drafters
    1560.: "15", # # .ENG-Surveying And Mapping Technicians
    1600.: "16", # # .SCI-Agricultural And Food Scientists
    1610.: "16", # # .SCI-Biological Scientists
    1640.: "16", # # .SCI-Conservation Scientists And Foresters
    1650.: "16", # # .SCI-Other Life Scientists
    1700.: "17", # # .SCI-Astronomers And Physicists
    1710.: "17", # # .SCI-Atmospheric And Space Scientists
    1720.: "17", # # .SCI-Chemists And Materials Scientists
    1745.: "17", # # .SCI-Environmental Scientists And Specialists, Including .Health
    1750.: "17", # # .SCI-Geoscientists And Hydrologists, Except Geographers
    1760.: "17", # # .SCI-Physical Scientists, All Other
    1800.: "18", # # .SCI-Economists
    1821.: "18", # # .SCI-Clinical And Counseling Psychologists
    1822.: "18", # # .SCI-School Psychologists
    1825.: "18", # # .SCI-Other Psychologists
    1840.: "18", # # .SCI-Urban And Regional Planners
    1860.: "18", # # .SCI-Other Social Scientists
    1900.: "19", # # .SCI-Agricultural And Food Science Technicians
    1910.: "19", # # .SCI-Biological Technicians
    1920.: "19", # # .SCI-Chemical Technicians
    1935.: "19", # # .SCI-Environmental Science and Geoscience Technicians, And .Nuclear Technicians
    1970.: "19", # # .SCI-Other Life, Physical, And Social Science Technicians
    1980.: "19", # # .SCI-Occupational Health And Safety Specialists and .Technicians
    2001.: "20", # # .CMS-Substance Abuse And Behavioral Disorder Counselors
    2002.: "20", # # .CMS-Educational, Guidance, And Career Counselors And .Advisors
    2003.: "20", # # .CMS-Marriage And Family Therapists
    2004.: "20", # # .CMS-Mental Health Counselors
    2005.: "20", # # .CMS-Rehabilitation Counselors
    2006.: "20", # # .CMS-Counselors, All Other
    2011.: "20", # # .CMS-Child, Family, And School Social Workers
    2012.: "20", # # .CMS-Healthcare Social Workers
    2013.: "20", # # .CMS-Mental Health And Substance Abuse Social Workers
    2014.: "20", # # .CMS-Social Workers, All Other
    2015.: "20", # # .CMS-Probation Officers And Correctional Treatment .Specialists
    2016.: "20", # # .CMS-Social And Human Service Assistants
    2025.: "20", # # .CMS-Other Community and Social Service Specialists
    2040.: "20", # # .CMS-Clergy
    2050.: "20", # # .CMS-Directors, Religious Activities And Education
    2060.: "20", # # .CMS-Religious Workers, All Other
    2100.: "21", # # .LGL-Lawyers, And Judges, Magistrates, And Other Judicial .Workers
    2105.: "21", # # .LGL-Judicial Law Clerks
    2145.: "21", # # .LGL-Paralegals And Legal Assistants
    2170.: "21", # # .LGL-Title Examiners, Abstractors, and Searchers
    2180.: "21", # # .LGL-Legal Support Workers, All Other
    2205.: "22", # # .EDU-Postsecondary Teachers
    2300.: "23", # # .EDU-Preschool And Kindergarten Teachers
    2310.: "23", # # .EDU-Elementary And Middle School Teachers
    2320.: "23", # # .EDU-Secondary School Teachers
    2330.: "23", # # .EDU-Special Education Teachers
    2350.: "23", # # .EDU-Tutors
    2360.: "23", # # .EDU-Other Teachers and Instructors
    2400.: "24", # # .EDU-Archivists, Curators, And Museum Technicians
    2430.: "24",  # Unknown (not in ACS data dict)
    2435.: "24", # # .EDU-Librarians And Media Collections Specialists
    2440.: "24", # # .EDU-Library Technicians
    2545.: "25", # # .EDU-Teaching Assistants
    2555.: "25", # # .EDU-Other Educational Instruction And Library Workers
    2600.: "26", # # .ENT-Artists And Related Workers
    2631.: "26", # # .ENT-Commercial And Industrial Designers
    2632.: "26", # # .ENT-Fashion Designers
    2633.: "26", # # .ENT-Floral Designers
    2634.: "26", # # .ENT-Graphic Designers
    2635.: "26", # # .ENT-Interior Designers
    2636.: "26", # # .ENT-Merchandise Displayers And Windows Trimmers
    2640.: "26", # # .ENT-Other Designers
    2700.: "27", # # .ENT-Actors
    2710.: "27", # # .ENT-Producers And Directors
    2721.: "27", # # .ENT-Athletes and Sports Competitors
    2722.: "27", # # .ENT-Coaches and Scouts
    2723.: "27", # # .ENT-Umpires, Referees, And Other Sports Officials
    2740.: "27", # # .ENT-Dancers And Choreographers
    2751.: "27", # # .ENT-Music Directors and Composers
    2752.: "27", # # .ENT-Musicians and Singers
    2755.: "27", # # .ENT-Disc Jockeys, Except Radio
    2770.: "27", # # .ENT-Entertainers And Performers, Sports and Related .Workers, All Other
    2805.: "28", # # .ENT-Broadcast Announcers And Radio Disc Jockeys
    2810.: "28", # # .ENT-News Analysts, Reporters And Correspondents
    2825.: "28", # # .ENT-Public Relations Specialists
    2830.: "28", # # .ENT-Editors
    2840.: "28", # # .ENT-Technical Writers
    2850.: "28", # # .ENT-Writers And Authors
    2861.: "28", # # .ENT-Interpreters and Translators
    2862.: "28", # # .ENT-Court Reporters and Simultaneous Captioners
    2865.: "28", # # .ENT-Media And Communication Workers, All Other
    2905.: "29", # # .ENT-Other Media And Communication Equipment Workers
    2910.: "29", # # .ENT-Photographers
    2920.: "29", # # .ENT-Television, Video, And Motion Picture Camera Operators .And Editors
    3000.: "30", # # .MED-Chiropractors
    3010.: "30", # # .MED-Dentists
    3030.: "30", # # .MED-Dietitians And Nutritionists
    3040.: "30", # # .MED-Optometrists
    3050.: "30", # # .MED-Pharmacists
    3090.: "30", # # .MED-Physicians
    3100.: "31", # # .MED-Surgeons
    3110.: "31", # # .MED-Physician Assistants
    3120.: "31", # # .MED-Podiatrists
    3140.: "31", # # .MED-Audiologists
    3150.: "31", # # .MED-Occupational Therapists
    3160.: "31", # # .MED-Physical Therapists
    3200.: "32", # # .MED-Radiation Therapists
    3210.: "32", # # .MED-Recreational Therapists
    3220.: "32", # # .MED-Respiratory Therapists
    3230.: "32", # # .MED-Speech-Language Pathologists
    3245.: "32", # # .MED-Other Therapists
    3250.: "32", # # .MED-Veterinarians
    3255.: "32", # # .MED-Registered Nurses
    3256.: "32", # # .MED-Nurse Anesthetists
    3258.: "32", # # .MED-Nurse Practitioners, And Nurse Midwives
    3261.: "32", # # .MED-Acupuncturists
    3270.: "32", # # .MED-Healthcare Diagnosing Or Treating Practitioners, All .Other
    3300.: "33", # # .MED-Clinical Laboratory Technologists And Technicians
    3310.: "33", # # .MED-Dental Hygienists
    3320.: "33", # # .MED-Unknown (not in ACS PUMS data dict)
    3321.: "33", # # .MED-Cardiovascular Technologists and Technicians
    3322.: "33", # # .MED-Diagnostic Medical Sonographers
    3323.: "33", # # .MED-Radiologic Technologists And Technicians
    3324.: "33", # # .MED-Magnetic Resonance Imaging Technologists
    3330.: "33", # # .MED-Nuclear Medicine Technologists and Medical Dosimetrists
    3401.: "34", # # .MED-Emergency Medical Technicians
    3402.: "34", # # .MED-Paramedics
    3421.: "34", # # .MED-Pharmacy Technicians
    3422.: "34", # # .MED-Psychiatric Technicians
    3423.: "34", # # .MED-Surgical Technologists
    3424.: "34", # # .MED-Veterinary Technologists and Technicians
    3430.: "34", # # .MED-Dietetic Technicians And Ophthalmic Medical Technicians
    3500.: "35", # # .MED-Licensed Practical And Licensed Vocational Nurses
    3515.: "35", # # .MED-Medical Records Specialists
    3520.: "35", # # .MED-Opticians, Dispensing
    3545.: "35", # # .MED-Miscellaneous Health Technologists and Technicians
    3550.: "35", # # .MED-Other Healthcare Practitioners and Technical .Occupations
    3601.: "36", # # .HLS-Home Health Aides
    3602.: "36", # # .HLS-Personal Care Aides
    3603.: "36", # # .HLS-Nursing Assistants
    3605.: "36", # # .HLS-Orderlies and Psychiatric Aides
    3610.: "36", # # .HLS-Occupational Therapy Assistants And Aides
    3620.: "36", # # .HLS-Physical Therapist Assistants And Aides
    3630.: "36", # # .HLS-Massage Therapists
    3640.: "36", # # .HLS-Dental Assistants
    3645.: "36", # # .HLS-Medical Assistants
    3646.: "36", # # .HLS-Medical Transcriptionists
    3647.: "36", # # .HLS-Pharmacy Aides
    3648.: "36", # # .HLS-Veterinary Assistants And Laboratory Animal Caretakers
    3649.: "36", # # .HLS-Phlebotomists
    3655.: "36", # # .HLS-Other Healthcare Support Workers
    3700.: "37", # # .PRT-First-Line Supervisors Of Correctional Officers
    3710.: "37", # # .PRT-First-Line Supervisors Of Police And Detectives
    3720.: "37", # # .PRT-First-Line Supervisors Of Fire Fighting And Prevention .Workers
    3725.: "37", # # .PRT-First-Line Supervisors of Security And Protective .Service Workers, All Other
    3740.: "37", # # .PRT-Firefighters
    3750.: "37", # # .PRT-Fire Inspectors
    3801.: "38", # # .PRT-Bailiffs
    3802.: "38", # # .PRT-Correctional Officers and Jailers
    3820.: "38", # # .PRT-Detectives And Criminal Investigators
    3840.: "38", # # .PRT-Fish And Game Wardens And Parking Enforcement Officers
    3870.: "38", # # .PRT-Police Officers
    3900.: "39", # # .PRT-Animal Control Workers
    3910.: "39", # # .PRT-Private Detectives And Investigators
    3930.: "39", # # .PRT-Security Guards And Gaming Surveillance Officers
    3940.: "39", # # .PRT-Crossing Guards And Flaggers
    3945.: "39", # # .PRT-Transportation Security Screeners
    3946.: "39", # # .PRT-School Bus Monitors
    3960.: "39", # # .PRT-Other Protective Service Workers
    4000.: "40", # # .EAT-Chefs And Head Cooks
    4010.: "40", # # .EAT-First-Line Supervisors Of Food Preparation And Serving .Workers
    4020.: "40", # # .EAT-Cooks
    4030.: "40", # # .EAT-Food Preparation Workers
    4040.: "40", # # .EAT-Bartenders
    4055.: "40", # # .EAT-Fast Food And Counter Workers
    4110.: "41", # # .EAT-Waiters And Waitresses
    4120.: "41", # # .EAT-Food Servers, Nonrestaurant
    4130.: "41", # # .EAT-Dining Room And Cafeteria Attendants And Bartender .Helpers
    4140.: "41", # # .EAT-Dishwashers
    4150.: "41", # # .EAT-Hosts And Hostesses, Restaurant, Lounge, And Coffee .Shop
    4160.: "41", # # .EAT-Food Preparation and Serving Related Workers, All Other
    4200.: "42", # # .CLN-First-Line Supervisors Of Housekeeping And Janitorial .Workers
    4210.: "42", # # .CLN-First-Line Supervisors Of Landscaping, Lawn Service, .And Groundskeeping Workers
    4220.: "42", # # .CLN-Janitors And Building Cleaners
    4230.: "42", # # .CLN-Maids And Housekeeping Cleaners
    4240.: "42", # # .CLN-Pest Control Workers
    4251.: "42", # # .CLN-Landscaping And Groundskeeping Workers
    4252.: "42", # # .CLN-Tree Trimmers and Pruners
    4255.: "42", # # .CLN-Other Grounds Maintenance Workers
    4330.: "43", # # .PRS-Supervisors Of Personal Care And Service Workers
    4340.: "43", # # .PRS-Animal Trainers
    4350.: "43", # # .PRS-Animal Caretakers
    4400.: "44", # # .PRS-Gambling Services Workers
    4420.: "44", # # .PRS-Ushers, Lobby Attendants, And Ticket Takers
    4435.: "44", # # .PRS-Other Entertainment Attendants And Related Workers
    4461.: "44", # # .PRS-Embalmers, Crematory Operators, And Funeral Attendants
    4465.: "44", # # .PRS-Morticians, Undertakers, And Funeral Arrangers
    4500.: "45", # # .PRS-Barbers
    4510.: "45", # # .PRS-Hairdressers, Hairstylists, And Cosmetologists
    4521.: "45", # # .PRS-Manicurists And Pedicurists
    4522.: "45", # # .PRS-Skincare Specialists
    4525.: "45", # # .PRS-Other Personal Appearance Workers
    4530.: "45", # # .PRS-Baggage Porters, Bellhops, And Concierges
    4540.: "45", # # .PRS-Tour And Travel Guides
    4600.: "46", # # .PRS-Childcare Workers
    4621.: "46", # # .PRS-Exercise Trainers And Group Fitness Instructors
    4622.: "46", # # .PRS-Recreation Workers
    4640.: "46", # # .PRS-Residential Advisors
    4655.: "46", # # .PRS-Personal Care and Service Workers, All Other
    4700.: "47", # # .SAL-First-Line Supervisors Of Retail Sales Workers
    4710.: "47", # # .SAL-First-Line Supervisors Of Non-Retail Sales Workers
    4720.: "47", # # .SAL-Cashiers
    4740.: "47", # # .SAL-Counter And Rental Clerks
    4750.: "47", # # .SAL-Parts Salespersons
    4760.: "47", # # .SAL-Retail Salespersons
    4800.: "48", # # .SAL-Advertising Sales Agents
    4810.: "48", # # .SAL-Insurance Sales Agents
    4820.: "48", # # .SAL-Securities, Commodities, And Financial Services Sales .Agents
    4830.: "48", # # .SAL-Travel Agents
    4840.: "48", # # .SAL-Sales Representatives Of Services, Except Advertising, .Insurance, Financial Services, And Travel
    4850.: "48", # # .SAL-Sales Representatives, Wholesale And Manufacturing
    4900.: "49", # # .SAL-Models, Demonstrators, And Product Promoters
    4920.: "49", # # .SAL-Real Estate Brokers And Sales Agents
    4930.: "49", # # .SAL-Sales Engineers
    4940.: "49", # # .SAL-Telemarketers
    4950.: "49", # # .SAL-Door-To-Door Sales Workers, News And Street Vendors, .And Related Workers
    4965.: "49", # # .SAL-Sales And Related Workers, All Other
    5000.: "50", # # .OFF-First-Line Supervisors Of Office And Administrative .Support Workers
    5010.: "50", # # .OFF-Switchboard Operators, Including Answering Service
    5020.: "50", # # .OFF-Telephone Operators
    5040.: "50", # # .OFF-Communications Equipment Operators, All Other
    5100.: "51", # # .OFF-Bill And Account Collectors
    5110.: "51", # # .OFF-Billing And Posting Clerks
    5120.: "51", # # .OFF-Bookkeeping, Accounting, And Auditing Clerks
    5140.: "51", # # .OFF-Payroll And Timekeeping Clerks
    5150.: "51", # # .OFF-Procurement Clerks
    5160.: "51", # # .OFF-Tellers
    5165.: "51", # # .OFF-Other Financial Clerks
    5220.: "52", # # .OFF-Court, Municipal, And License Clerks
    5230.: "52", # # .OFF-Credit Authorizers, Checkers, And Clerks
    5240.: "52", # # .OFF-Customer Service Representatives
    5250.: "52", # # .OFF-Eligibility Interviewers, Government Programs
    5260.: "52", # # .OFF-File Clerks
    5300.: "53", # # .OFF-Hotel, Motel, And Resort Desk Clerks
    5310.: "53", # # .OFF-Interviewers, Except Eligibility And Loan
    5320.: "53", # # .OFF-Library Assistants, Clerical
    5330.: "53", # # .OFF-Loan Interviewers And Clerks
    5340.: "53", # # .OFF-New Accounts Clerks
    5350.: "53", # # .OFF-Correspondence Clerks And Order Clerks
    5360.: "53", # # .OFF-Human Resources Assistants, Except Payroll And .Timekeeping
    5400.: "54", # # .OFF-Receptionists And Information Clerks
    5410.: "54", # # .OFF-Reservation And Transportation Ticket Agents And Travel .Clerks
    5420.: "54", # # .OFF-Other Information And Records Clerks
    5500.: "55", # # .OFF-Cargo And Freight Agents
    5510.: "55", # # .OFF-Couriers And Messengers
    5521.: "55", # # .OFF-Public Safety Telecommunicators
    5522.: "55", # # .OFF-Dispatchers, Except Police, Fire, And Ambulance
    5530.: "55", # # .OFF-Meter Readers, Utilities
    5540.: "55", # # .OFF-Postal Service Clerks
    5550.: "55", # # .OFF-Postal Service Mail Carriers
    5560.: "55", # # .OFF-Postal Service Mail Sorters, Processors, And Processing .Machine Operators
    5600.: "56", # # .OFF-Production, Planning, And Expediting Clerks
    5610.: "56", # # .OFF-Shipping, Receiving, And Inventory Clerks
    5630.: "56", # # .OFF-Weighers, Measurers, Checkers, And Samplers, .Recordkeeping
    5710.: "57", # # .OFF-Executive Secretaries And Executive Administrative .Assistants
    5720.: "57", # # .OFF-Legal Secretaries and Administrative Assistants
    5730.: "57", # # .OFF-Medical Secretaries and Administrative Assistants
    5740.: "57", # # .OFF-Secretaries And Administrative Assistants, Except .Legal, Medial, And Executive
    5810.: "58", # # .OFF-Data Entry Keyers
    5820.: "58", # # .OFF-Word Processors And Typists
    5840.: "58", # # .OFF-Insurance Claims And Policy Processing Clerks
    5850.: "58", # # .OFF-Mail Clerks And Mail Machine Operators, Except Postal .Service
    5860.: "58", # # .OFF-Office Clerks, General
    5900.: "59", # # .OFF-Office Machine Operators, Except Computer
    5910.: "59", # # .OFF-Proofreaders And Copy Markers
    5920.: "59", # # .OFF-Statistical Assistants
    5940.: "59", # # .OFF-Other Office And Administrative Support Workers
    6005.: "60", # # .FFF-First-Line Supervisors Of Farming, Fishing, And .Forestry Workers
    6010.: "60", # # .FFF-Agricultural Inspectors
    6040.: "60", # # .FFF-Graders And Sorters, Agricultural Products
    6050.: "60", # # .FFF-Other Agricultural Workers
    6115.: "61", # # .FFF-Fishing And Hunting Workers
    6120.: "61", # # .FFF-Forest And Conservation Workers
    6130.: "61", # # .FFF-Logging Workers
    6200.: "62", # # .CON-First-Line Supervisors Of Construction Trades And .Extraction Workers
    6210.: "62", # # .CON-Boilermakers
    6220.: "62", # # .CON-Brickmasons, Blockmasons, Stonemasons, And Reinforcing .Iron And Rebar Workers
    6230.: "62", # # .CON-Carpenters
    6240.: "62", # # .CON-Carpet, Floor, And Tile Installers And Finishers
    6250.: "62", # # .CON-Cement Masons, Concrete Finishers, And Terrazzo Workers
    6260.: "62", # # .CON-Construction Laborers
    6305.: "63", # # .CON-Construction Equipment Operators
    6330.: "63", # # .CON-Drywall Installers, Ceiling Tile Installers, And Tapers
    6355.: "63", # # .CON-Electricians
    6360.: "63", # # .CON-Glaziers
    6400.: "64", # # .CON-Insulation Workers
    6410.: "64", # # .CON-Painters and Paperhangers
    6441.: "64", # # .CON-Pipelayers
    6442.: "64", # # .CON-Plumbers, Pipefitters, And Steamfitters
    6460.: "64", # # .CON-Plasterers And Stucco Masons
    6515.: "65", # # .CON-Roofers
    6520.: "65", # # .CON-Sheet Metal Workers
    6530.: "65", # # .CON-Structural Iron And Steel Workers
    6540.: "65", # # .CON-Solar Photovoltaic Installers
    6600.: "66", # # .CON-Helpers, Construction Trades
    6660.: "66", # # .CON-Construction And Building Inspectors
    6700.: "67", # # .CON-Elevator Installers And Repairers
    6710.: "67", # # .CON-Fence Erectors
    6720.: "67", # # .CON-Hazardous Materials Removal Workers
    6730.: "67", # # .CON-Highway Maintenance Workers
    6740.: "67", # # .CON-Rail-Track Laying And Maintenance Equipment Operators
    6765.: "67", # # .CON-Other Construction And Related Workers
    6800.: "68", # # .EXT-Derrick, Rotary Drill, And Service Unit Operators, And .Roustabouts, Oil, Gas, And Mining
    6825.: "68", # # .EXT-Surface Mining Machine Operators And Earth Drillers
    6835.: "68", # # .EXT-Explosives Workers, Ordnance Handling Experts, and .Blasters
    6850.: "68", # # .EXT-Underground Mining Machine Operators
    6950.: "69", # # .EXT-Other Extraction Workers
    7000.: "70", # # .RPR-First-Line Supervisors Of Mechanics, Installers, And .Repairers
    7010.: "70", # # .RPR-Computer, Automated Teller, And Office Machine .Repairers
    7020.: "70", # # .RPR-Radio And Telecommunications Equipment Installers And .Repairers
    7030.: "70", # # .RPR-Avionics Technicians
    7040.: "70", # # .RPR-Electric Motor, Power Tool, And Related Repairers
    7100.: "71", # # .RPR-Other Electrical And Electronic Equipment Mechanics, .Installers, And Repairers.
    7120.: "71", # # .RPR-Electronic Home Entertainment Equipment Installers And .Repairers
    7130.: "71", # # .RPR-Security And Fire Alarm Systems Installers
    7140.: "71", # # .RPR-Aircraft Mechanics And Service Technicians
    7150.: "71", # # .RPR-Automotive Body And Related Repairers
    7160.: "71", # # .RPR-Automotive Glass Installers And Repairers
    7200.: "72", # # .RPR-Automotive Service Technicians And Mechanics
    7210.: "72", # # .RPR-Bus And Truck Mechanics And Diesel Engine Specialists
    7220.: "72", # # .RPR-Heavy Vehicle And Mobile Equipment Service Technicians .And Mechanics
    7240.: "72", # # .RPR-Small Engine Mechanics
    7260.: "72", # # .RPR-Miscellaneous Vehicle And Mobile Equipment Mechanics, .Installers, And Repairers
    7300.: "73", # # .RPR-Control And Valve Installers And Repairers
    7315.: "73", # # .RPR-Heating, Air Conditioning, And Refrigeration Mechanics .And Installers
    7320.: "73", # # .RPR-Home Appliance Repairers
    7330.: "73", # # .RPR-Industrial And Refractory Machinery Mechanics
    7340.: "73", # # .RPR-Maintenance And Repair Workers, General
    7350.: "73", # # .RPR-Maintenance Workers, Machinery
    7360.: "73", # # .RPR-Millwrights
    7410.: "74", # # .RPR-Electrical Power-Line Installers And Repairers
    7420.: "74", # # .RPR-Telecommunications Line Installers And Repairers
    7430.: "74", # # .RPR-Precision Instrument And Equipment Repairers
    7510.: "75", # # .RPR-Coin, Vending, And Amusement Machine Servicers And .Repairers
    7540.: "75", # # .RPR-Locksmiths And Safe Repairers
    7560.: "75", # # .RPR-Riggers
    7610.: "76", # # .RPR-Helpers--Installation, Maintenance, And Repair Workers
    7640.: "76", # # .RPR-Other Installation, Maintenance, And Repair Workers
    7700.: "77", # # .PRD-First-Line Supervisors Of Production And Operating .Workers
    7720.: "77", # # .PRD-Electrical, Electronics, And Electromechanical .Assemblers
    7730.: "77", # # .PRD-Engine And Other Machine Assemblers
    7740.: "77", # # .PRD-Structural Metal Fabricators And Fitters
    7750.: "77", # # .PRD-Other Assemblers And Fabricators
    7800.: "78", # # .PRD-Bakers
    7810.: "78", # # .PRD-Butchers And Other Meat, Poultry, And Fish Processing .Workers
    7830.: "78", # # .PRD-Food And Tobacco Roasting, Baking, And Drying Machine .Operators And Tenders
    7840.: "78", # # .PRD-Food Batchmakers
    7850.: "78", # # .PRD-Food Cooking Machine Operators And Tenders
    7855.: "78", # # .PRD-Food Processing Workers, All Other
    7905.: "79", # # .PRD-Computer Numerically Controlled Tool Operators And .Programmers
    7925.: "79", # # .PRD-Forming Machine Setters, Operators, And Tenders, Metal .And Plastic
    7950.: "79", # # .PRD-Cutting, Punching, And Press Machine Setters, .Operators, And Tenders, Metal And Plastic
    8000.: "80", # # .Grinding, Lapping, Polishing, And Buffing Machine Tool
    8025.: "80", # # .PRD-Other Machine Tool Setters, Operators, And Tenders, .Metal and Plastic
    8030.: "80", # # .PRD-Machinists
    8040.: "80", # # .PRD-Metal Furnace Operators, Tenders, Pourers, And Casters
    8100.: "81", # # .PRD-Model Makers, Patternmakers, And Molding Machine .Setters, Metal And Plastic
    8130.: "81", # # .PRD-Tool And Die Makers
    8140.: "81", # # .PRD-Welding, Soldering, And Brazing Workers
    8225.: "82", # # .PRD-Other Metal Workers And Plastic Workers
    8250.: "82", # # .PRD-Prepress Technicians And Workers
    8255.: "82", # # .PRD-Printing Press Operators
    8256.: "82", # # .PRD-Print Binding And Finishing Workers
    8300.: "83", # # .PRD-Laundry And Dry-Cleaning Workers
    8310.: "83", # # .PRD-Pressers, Textile, Garment, And Related Materials
    8320.: "83", # # .PRD-Sewing Machine Operators
    8335.: "83", # # .PRD-Shoe And Leather Workers
    8350.: "83", # # .PRD-Tailors, Dressmakers, And Sewers
    8365.: "83", # # .PRD-Textile Machine Setters, Operators, And Tenders
    8410.: "84",  # .PRD - Unknown (not in ACS PUMS data dictionary)
    8450.: "84", # # .PRD-Upholsterers
    8465.: "84", # # .PRD-Other Textile, Apparel, And Furnishings Workers
    8500.: "85", # # .PRD-Cabinetmakers And Bench Carpenters
    8510.: "85", # # .PRD-Furniture Finishers
    8530.: "85", # # .PRD-Sawing Machine Setters, Operators, And Tenders, Wood
    8540.: "85", # # .PRD-Woodworking Machine Setters, Operators, And Tenders, .Except Sawing
    8555.: "85", # # .PRD-Other Woodworkers
    8600.: "86", # # .PRD-Power Plant Operators, Distributors, And Dispatchers
    8610.: "86", # # .PRD-Stationary Engineers And Boiler Operators
    8620.: "86", # # .PRD-Water And Wastewater Treatment Plant And System .Operators
    8630.: "86", # # .PRD-Miscellaneous Plant And System Operators
    8640.: "86", # # .PRD-Chemical Processing Machine Setters, Operators, And .Tenders
    8650.: "86", # # .PRD-Crushing, Grinding, Polishing, Mixing, And Blending .Workers
    8710.: "87", # # .PRD-Cutting Workers
    8720.: "87", # # .PRD-Extruding, Forming, Pressing, And Compacting Machine .Setters, Operators, And Tenders
    8730.: "87", # # .PRD-Furnace, Kiln, Oven, Drier, And Kettle Operators And .Tenders
    8740.: "87", # # .PRD-Inspectors, Testers, Sorters, Samplers, And Weighers
    8750.: "87", # # .PRD-Jewelers And Precious Stone And Metal Workers
    8760.: "87", # # .PRD-Dental And Ophthalmic Laboratory Technicians And .Medical Appliance Technicians
    8800.: "88", # # .PRD-Packaging And Filling Machine Operators And Tenders
    8810.: "88", # # .PRD-Painting Workers
    8830.: "88", # # .PRD-Photographic Process Workers And Processing Machine .Operators
    8850.: "88", # # .PRD-Adhesive Bonding Machine Operators And Tenders
    8910.: "89", # # .PRD-Etchers And Engravers
    8920.: "89", # # .PRD-Molders, Shapers, And Casters, Except Metal And Plastic
    8930.: "89", # # .PRD-Paper Goods Machine Setters, Operators, And Tenders
    8940.: "89", # # .PRD-Tire Builders
    8950.: "89", # # .PRD-Helpers-Production Workers
    8990.: "89", # # .PRD-Miscellaneous Production Workers, Including Equipment .Operators And Tenders
    9005.: "90", # # .TRN-Supervisors Of Transportation And Material Moving .Workers
    9030.: "90", # # .TRN-Aircraft Pilots And Flight Engineers
    9040.: "90", # # .TRN-Air Traffic Controllers And Airfield Operations .Specialists
    9050.: "90", # # .TRN-Flight Attendants
    9110.: "91", # # .TRN-Ambulance Drivers And Attendants, Except Emergency .Medical Technicians
    9121.: "91", # # .TRN-Bus Drivers, School
    9122.: "91", # # .TRN-Bus Drivers, Transit And Intercity
    9130.: "91", # # .TRN-Driver/Sales Workers And Truck Drivers
    9141.: "91", # # .TRN-Shuttle Drivers And Chauffeurs
    9142.: "91", # # .TRN-Taxi Drivers
    9150.: "91", # # .TRN-Motor Vehicle Operators, All Other
    9210.: "92", # # .TRN-Locomotive Engineers And Operators
    9240.: "92", # # .TRN-Railroad Conductors And Yardmasters
    9265.: "92", # # .TRN-Other Rail Transportation Workers
    9300.: "93", # # .TRN-Sailors And Marine Oilers, And Ship Engineers
    9310.: "93", # # .TRN-Ship And Boat Captains And Operators
    9350.: "93", # # .TRN-Parking Lot Attendants
    9365.: "93", # # .TRN-Transportation Service Attendants
    9410.: "94", # # .TRN-Transportation Inspectors
    9415.: "94", # # .TRN-Passenger Attendants
    9430.: "94", # # .TRN-Other Transportation Workers
    9510.: "95", # # .TRN-Crane And Tower Operators
    9570.: "95", # # .TRN-Conveyor, Dredge, And Hoist and Winch Operators
    9600.: "96", # # .TRN-Industrial Truck And Tractor Operators
    9610.: "96", # # .TRN-Cleaners Of Vehicles And Equipment
    9620.: "96", # # .TRN-Laborers And Freight, Stock, And Material Movers, Hand
    9630.: "96", # # .TRN-Machine Feeders And Offbearers
    9640.: "96", # # .TRN-Packers And Packagers, Hand
    9645.: "96", # # .TRN-Stockers And Order Fillers
    9650.: "96", # # .TRN-Pumping Station Operators
    9720.: "97", # # .TRN-Refuse And Recyclable Material Collectors
    9760.: "97", # # .TRN-Other Material Moving Workers
    9800.: "98", # # .MIL-Military Officer Special And Tactical Operations .Leaders
    9810.: "98", # # .MIL-First-Line Enlisted Military Supervisors
    9825.: "98", # # .MIL-Military Enlisted Tactical Operations And Air/Weapons .Specialists And Crew Members
    9830.: "98", # # .MIL-Military, Rank Not Specified
    9920.: "99", # # .Unemployed And Last Worked 5 Years Ago Or Earlier Or Never .Worked
    1020.: "UNK",  # Unknown category not documented in ACS PUMS documentation.
}



OCCP_MAPPING_COARSE = {
    # "bbbb": "", # .N/A (less than 16 years old/NILF who last worked more than .5 years ago or never worked)
    0010.: "MGR", # .MGR-Chief Executives And Legislators
    0020.: "MGR", # .MGR-General And Operations Managers
    0040.: "MGR", # .MGR-Advertising And Promotions Managers
    0051.: "MGR", # .MGR-Marketing Managers
    0052.: "MGR", # .MGR-Sales Managers
    0060.: "MGR", # .MGR-Public Relations And Fundraising Managers
    0101.: "MGR", # .MGR-Administrative Services Managers
    0102.: "MGR", # .MGR-Facilities Managers
    0110.: "MGR", # .MGR-Computer And Information Systems Managers
    0120.: "MGR", # .MGR-Financial Managers
    0135.: "MGR", # .MGR-Compensation And Benefits Managers
    0136.: "MGR", # .MGR-Human Resources Managers
    0137.: "MGR", # .MGR-Training And Development Managers
    0140.: "MGR", # .MGR-Industrial Production Managers
    0150.: "MGR", # .MGR-Purchasing Managers
    0160.: "MGR", # .MGR-Transportation, Storage, And Distribution Managers
    0205.: "MGR", # .MGR-Farmers, Ranchers, And Other Agricultural Managers
    0220.: "MGR", # .MGR-Construction Managers
    0230.: "MGR", # .MGR-Education And Childcare Administrators
    0300.: "MGR", # .MGR-Architectural And Engineering Managers
    0310.: "MGR", # .MGR-Food Service Managers
    0335.: "MGR", # .MGR-Entertainment and Recreation Managers
    0340.: "MGR", # .MGR-Lodging Managers
    0350.: "MGR", # .MGR-Medical And Health Services Managers
    0360.: "MGR", # .MGR-Natural Sciences Managers
    0410.: "MGR", # .MGR-Property, Real Estate, And Community Association .Managers
    0420.: "MGR", # .MGR-Social And Community Service Managers
    0425.: "MGR", # .MGR-Emergency Management Directors
    0440.: "MGR", # .MGR-Other Managers
    0500.: "BUS", # .BUS-Agents And Business Managers Of Artists, Performers, .And Athletes
    0510.: "BUS", # .BUS-Buyers And Purchasing Agents, Farm Products
    0520.: "BUS", # .BUS-Wholesale And Retail Buyers, Except Farm Products
    0530.: "BUS", # .BUS-Purchasing Agents, Except Wholesale, Retail, And Farm .Products
    0540.: "BUS", # .BUS-Claims Adjusters, Appraisers, Examiners, And .Investigators
    0565.: "BUS", # .BUS-Compliance Officers
    0600.: "BUS", # .BUS-Cost Estimators
    0630.: "BUS", # .BUS-Human Resources Workers
    0640.: "BUS", # .BUS-Compensation, Benefits, And Job Analysis Specialists
    0650.: "BUS", # .BUS-Training And Development Specialists
    0700.: "BUS", # .BUS-Logisticians
    0705.: "BUS", # .BUS-Project Management Specialists
    0710.: "BUS", # .BUS-Management Analysts
    0725.: "BUS", # .BUS-Meeting, Convention, And Event Planners
    0726.: "BUS", # .BUS-Fundraisers
    0735.: "BUS", # .BUS-Market Research Analysts And Marketing Specialists
    0750.: "BUS", # .BUS-Business Operations Specialists, All Other
    0800.: "FIN", # .FIN-Accountants And Auditors
    0810.: "FIN", # .FIN-Property Appraisers and Assessors
    0820.: "FIN", # .FIN-Budget Analysts
    0830.: "FIN", # .FIN-Credit Analysts
    0845.: "FIN", # .FIN-Financial And Investment Analysts
    0850.: "FIN", # .FIN-Personal Financial Advisors
    0860.: "FIN", # .FIN-Insurance Underwriters
    0900.: "FIN", # .FIN-Financial Examiners
    0910.: "FIN", # .FIN-Credit Counselors And Loan Officers
    0930.: "FIN", # .FIN-Tax Examiners And Collectors, And Revenue Agents
    0940.: "FIN", # .FIN-Tax Preparers
    0960.: "FIN", # .FIN-Other Financial Specialists
    1005.: "CMM", # .CMM-Computer And Information Research Scientists
    1006.: "CMM", # .CMM-Computer Systems Analysts
    1007.: "CMM", # .CMM-Information Security Analysts
    1010.: "CMM", # .CMM-Computer Programmers
    1021.: "CMM", # .CMM-Software Developers
    1022.: "CMM", # .CMM-Software Quality Assurance Analysts and Testers
    1031.: "CMM", # .CMM-Web Developers
    1032.: "CMM", # .CMM-Web And Digital Interface Designers
    1050.: "CMM", # .CMM-Computer Support Specialists
    1065.: "CMM", # .CMM-Database Administrators and Architects
    1105.: "CMM", # .CMM-Network And Computer Systems Administrators
    1106.: "CMM", # .CMM-Computer Network Architects
    1108.: "CMM", # .CMM-Computer Occupations, All Other
    1200.: "CMM", # .CMM-Actuaries
    1220.: "CMM", # .CMM-Operations Research Analysts
    1240.: "CMM", # .CMM-Other Mathematical Science Occupations
    1305.: "ENG", # .ENG-Architects, Except Landscape And Naval
    1306.: "ENG", # .ENG-Landscape Architects
    1310.: "ENG", # .ENG-Surveyors, Cartographers, And Photogrammetrists
    1320.: "ENG", # .ENG-Aerospace Engineers
    1340.: "ENG", # .ENG-Biomedical And Agricultural Engineers
    1350.: "ENG", # .ENG-Chemical Engineers
    1360.: "ENG", # .ENG-Civil Engineers
    1400.: "ENG", # .ENG-Computer Hardware Engineers
    1410.: "ENG", # .ENG-Electrical And Electronics Engineers
    1420.: "ENG", # .ENG-Environmental Engineers
    1430.: "ENG", # .ENG-Industrial Engineers, Including Health And Safety
    1440.: "ENG", # .ENG-Marine Engineers And Naval Architects
    1450.: "ENG", # .ENG-Materials Engineers
    1460.: "ENG", # .ENG-Mechanical Engineers
    1520.: "ENG", # .ENG-Petroleum, Mining And Geological Engineers, Including .Mining Safety Engineers
    1530.: "ENG", # .ENG-Other Engineers
    1541.: "ENG", # .ENG-Architectural And Civil Drafters
    1545.: "ENG", # .ENG-Other Drafters
    1551.: "ENG", # .ENG-Electrical And Electronic Engineering Technologists and .Technicians
    1555.: "Oth", # .Other Engineering Technologists And Technicians, Except .Drafters
    1560.: "ENG", # .ENG-Surveying And Mapping Technicians
    1600.: "SCI", # .SCI-Agricultural And Food Scientists
    1610.: "SCI", # .SCI-Biological Scientists
    1640.: "SCI", # .SCI-Conservation Scientists And Foresters
    1650.: "SCI", # .SCI-Other Life Scientists
    1700.: "SCI", # .SCI-Astronomers And Physicists
    1710.: "SCI", # .SCI-Atmospheric And Space Scientists
    1720.: "SCI", # .SCI-Chemists And Materials Scientists
    1745.: "SCI", # .SCI-Environmental Scientists And Specialists, Including .Health
    1750.: "SCI", # .SCI-Geoscientists And Hydrologists, Except Geographers
    1760.: "SCI", # .SCI-Physical Scientists, All Other
    1800.: "SCI", # .SCI-Economists
    1821.: "SCI", # .SCI-Clinical And Counseling Psychologists
    1822.: "SCI", # .SCI-School Psychologists
    1825.: "SCI", # .SCI-Other Psychologists
    1840.: "SCI", # .SCI-Urban And Regional Planners
    1860.: "SCI", # .SCI-Other Social Scientists
    1900.: "SCI", # .SCI-Agricultural And Food Science Technicians
    1910.: "SCI", # .SCI-Biological Technicians
    1920.: "SCI", # .SCI-Chemical Technicians
    1935.: "SCI", # .SCI-Environmental Science and Geoscience Technicians, And .Nuclear Technicians
    1970.: "SCI", # .SCI-Other Life, Physical, And Social Science Technicians
    1980.: "SCI", # .SCI-Occupational Health And Safety Specialists and .Technicians
    2001.: "CMS", # .CMS-Substance Abuse And Behavioral Disorder Counselors
    2002.: "CMS", # .CMS-Educational, Guidance, And Career Counselors And .Advisors
    2003.: "CMS", # .CMS-Marriage And Family Therapists
    2004.: "CMS", # .CMS-Mental Health Counselors
    2005.: "CMS", # .CMS-Rehabilitation Counselors
    2006.: "CMS", # .CMS-Counselors, All Other
    2011.: "CMS", # .CMS-Child, Family, And School Social Workers
    2012.: "CMS", # .CMS-Healthcare Social Workers
    2013.: "CMS", # .CMS-Mental Health And Substance Abuse Social Workers
    2014.: "CMS", # .CMS-Social Workers, All Other
    2015.: "CMS", # .CMS-Probation Officers And Correctional Treatment .Specialists
    2016.: "CMS", # .CMS-Social And Human Service Assistants
    2025.: "CMS", # .CMS-Other Community and Social Service Specialists
    2040.: "CMS", # .CMS-Clergy
    2050.: "CMS", # .CMS-Directors, Religious Activities And Education
    2060.: "CMS", # .CMS-Religious Workers, All Other
    2100.: "LGL", # .LGL-Lawyers, And Judges, Magistrates, And Other Judicial .Workers
    2105.: "LGL", # .LGL-Judicial Law Clerks
    2145.: "LGL", # .LGL-Paralegals And Legal Assistants
    2170.: "LGL", # .LGL-Title Examiners, Abstractors, and Searchers
    2180.: "LGL", # .LGL-Legal Support Workers, All Other
    2205.: "EDU", # .EDU-Postsecondary Teachers
    2300.: "EDU", # .EDU-Preschool And Kindergarten Teachers
    2310.: "EDU", # .EDU-Elementary And Middle School Teachers
    2320.: "EDU", # .EDU-Secondary School Teachers
    2330.: "EDU", # .EDU-Special Education Teachers
    2350.: "EDU", # .EDU-Tutors
    2360.: "EDU", # .EDU-Other Teachers and Instructors
    2400.: "EDU", # .EDU-Archivists, Curators, And Museum Technicians
    2430.: "EDU", # Unknown (not in ACS data dict)
    2435.: "EDU", # .EDU-Librarians And Media Collections Specialists
    2440.: "EDU", # .EDU-Library Technicians
    2545.: "EDU", # .EDU-Teaching Assistants
    2555.: "EDU", # .EDU-Other Educational Instruction And Library Workers
    2600.: "ENT", # .ENT-Artists And Related Workers
    2631.: "ENT", # .ENT-Commercial And Industrial Designers
    2632.: "ENT", # .ENT-Fashion Designers
    2633.: "ENT", # .ENT-Floral Designers
    2634.: "ENT", # .ENT-Graphic Designers
    2635.: "ENT", # .ENT-Interior Designers
    2636.: "ENT", # .ENT-Merchandise Displayers And Windows Trimmers
    2640.: "ENT", # .ENT-Other Designers
    2700.: "ENT", # .ENT-Actors
    2710.: "ENT", # .ENT-Producers And Directors
    2721.: "ENT", # .ENT-Athletes and Sports Competitors
    2722.: "ENT", # .ENT-Coaches and Scouts
    2723.: "ENT", # .ENT-Umpires, Referees, And Other Sports Officials
    2740.: "ENT", # .ENT-Dancers And Choreographers
    2751.: "ENT", # .ENT-Music Directors and Composers
    2752.: "ENT", # .ENT-Musicians and Singers
    2755.: "ENT", # .ENT-Disc Jockeys, Except Radio
    2770.: "ENT", # .ENT-Entertainers And Performers, Sports and Related .Workers, All Other
    2805.: "ENT", # .ENT-Broadcast Announcers And Radio Disc Jockeys
    2810.: "ENT", # .ENT-News Analysts, Reporters And Correspondents
    2825.: "ENT", # .ENT-Public Relations Specialists
    2830.: "ENT", # .ENT-Editors
    2840.: "ENT", # .ENT-Technical Writers
    2850.: "ENT", # .ENT-Writers And Authors
    2861.: "ENT", # .ENT-Interpreters and Translators
    2862.: "ENT", # .ENT-Court Reporters and Simultaneous Captioners
    2865.: "ENT", # .ENT-Media And Communication Workers, All Other
    2905.: "ENT", # .ENT-Other Media And Communication Equipment Workers
    2910.: "ENT", # .ENT-Photographers
    2920.: "ENT", # .ENT-Television, Video, And Motion Picture Camera Operators .And Editors
    3000.: "MED", # .MED-Chiropractors
    3010.: "MED", # .MED-Dentists
    3030.: "MED", # .MED-Dietitians And Nutritionists
    3040.: "MED", # .MED-Optometrists
    3050.: "MED", # .MED-Pharmacists
    3090.: "MED", # .MED-Physicians
    3100.: "MED", # .MED-Surgeons
    3110.: "MED", # .MED-Physician Assistants
    3120.: "MED", # .MED-Podiatrists
    3140.: "MED", # .MED-Audiologists
    3150.: "MED", # .MED-Occupational Therapists
    3160.: "MED", # .MED-Physical Therapists
    3200.: "MED", # .MED-Radiation Therapists
    3210.: "MED", # .MED-Recreational Therapists
    3220.: "MED", # .MED-Respiratory Therapists
    3230.: "MED", # .MED-Speech-Language Pathologists
    3245.: "MED", # .MED-Other Therapists
    3250.: "MED", # .MED-Veterinarians
    3255.: "MED", # .MED-Registered Nurses
    3256.: "MED", # .MED-Nurse Anesthetists
    3258.: "MED", # .MED-Nurse Practitioners, And Nurse Midwives
    3261.: "MED", # .MED-Acupuncturists
    3270.: "MED", # .MED-Healthcare Diagnosing Or Treating Practitioners, All .Other
    3300.: "MED", # .MED-Clinical Laboratory Technologists And Technicians
    3310.: "MED", # .MED-Dental Hygienists
    3320.: "MED", # .MED-Unknown (not in ACS PUMPS data dict)
    3321.: "MED", # .MED-Cardiovascular Technologists and Technicians
    3322.: "MED", # .MED-Diagnostic Medical Sonographers
    3323.: "MED", # .MED-Radiologic Technologists And Technicians
    3324.: "MED", # .MED-Magnetic Resonance Imaging Technologists
    3330.: "MED", # .MED-Nuclear Medicine Technologists and Medical Dosimetrists
    3401.: "MED", # .MED-Emergency Medical Technicians
    3402.: "MED", # .MED-Paramedics
    3421.: "MED", # .MED-Pharmacy Technicians
    3422.: "MED", # .MED-Psychiatric Technicians
    3423.: "MED", # .MED-Surgical Technologists
    3424.: "MED", # .MED-Veterinary Technologists and Technicians
    3430.: "MED", # .MED-Dietetic Technicians And Ophthalmic Medical Technicians
    3500.: "MED", # .MED-Licensed Practical And Licensed Vocational Nurses
    3515.: "MED", # .MED-Medical Records Specialists
    3520.: "MED", # .MED-Opticians, Dispensing
    3545.: "MED", # .MED-Miscellaneous Health Technologists and Technicians
    3550.: "MED", # .MED-Other Healthcare Practitioners and Technical .Occupations
    3601.: "HLS", # .HLS-Home Health Aides
    3602.: "HLS", # .HLS-Personal Care Aides
    3603.: "HLS", # .HLS-Nursing Assistants
    3605.: "HLS", # .HLS-Orderlies and Psychiatric Aides
    3610.: "HLS", # .HLS-Occupational Therapy Assistants And Aides
    3620.: "HLS", # .HLS-Physical Therapist Assistants And Aides
    3630.: "HLS", # .HLS-Massage Therapists
    3640.: "HLS", # .HLS-Dental Assistants
    3645.: "HLS", # .HLS-Medical Assistants
    3646.: "HLS", # .HLS-Medical Transcriptionists
    3647.: "HLS", # .HLS-Pharmacy Aides
    3648.: "HLS", # .HLS-Veterinary Assistants And Laboratory Animal Caretakers
    3649.: "HLS", # .HLS-Phlebotomists
    3655.: "HLS", # .HLS-Other Healthcare Support Workers
    3700.: "PRT", # .PRT-First-Line Supervisors Of Correctional Officers
    3710.: "PRT", # .PRT-First-Line Supervisors Of Police And Detectives
    3720.: "PRT", # .PRT-First-Line Supervisors Of Fire Fighting And Prevention .Workers
    3725.: "PRT", # .PRT-First-Line Supervisors of Security And Protective .Service Workers, All Other
    3740.: "PRT", # .PRT-Firefighters
    3750.: "PRT", # .PRT-Fire Inspectors
    3801.: "PRT", # .PRT-Bailiffs
    3802.: "PRT", # .PRT-Correctional Officers and Jailers
    3820.: "PRT", # .PRT-Detectives And Criminal Investigators
    3840.: "PRT", # .PRT-Fish And Game Wardens And Parking Enforcement Officers
    3870.: "PRT", # .PRT-Police Officers
    3900.: "PRT", # .PRT-Animal Control Workers
    3910.: "PRT", # .PRT-Private Detectives And Investigators
    3930.: "PRT", # .PRT-Security Guards And Gaming Surveillance Officers
    3940.: "PRT", # .PRT-Crossing Guards And Flaggers
    3945.: "PRT", # .PRT-Transportation Security Screeners
    3946.: "PRT", # .PRT-School Bus Monitors
    3960.: "PRT", # .PRT-Other Protective Service Workers
    4000.: "EAT", # .EAT-Chefs And Head Cooks
    4010.: "EAT", # .EAT-First-Line Supervisors Of Food Preparation And Serving .Workers
    4020.: "EAT", # .EAT-Cooks
    4030.: "EAT", # .EAT-Food Preparation Workers
    4040.: "EAT", # .EAT-Bartenders
    4055.: "EAT", # .EAT-Fast Food And Counter Workers
    4110.: "EAT", # .EAT-Waiters And Waitresses
    4120.: "EAT", # .EAT-Food Servers, Nonrestaurant
    4130.: "EAT", # .EAT-Dining Room And Cafeteria Attendants And Bartender .Helpers
    4140.: "EAT", # .EAT-Dishwashers
    4150.: "EAT", # .EAT-Hosts And Hostesses, Restaurant, Lounge, And Coffee .Shop
    4160.: "EAT", # .EAT-Food Preparation and Serving Related Workers, All Other
    4200.: "CLN", # .CLN-First-Line Supervisors Of Housekeeping And Janitorial .Workers
    4210.: "CLN", # .CLN-First-Line Supervisors Of Landscaping, Lawn Service, .And Groundskeeping Workers
    4220.: "CLN", # .CLN-Janitors And Building Cleaners
    4230.: "CLN", # .CLN-Maids And Housekeeping Cleaners
    4240.: "CLN", # .CLN-Pest Control Workers
    4251.: "CLN", # .CLN-Landscaping And Groundskeeping Workers
    4252.: "CLN", # .CLN-Tree Trimmers and Pruners
    4255.: "CLN", # .CLN-Other Grounds Maintenance Workers
    4330.: "PRS", # .PRS-Supervisors Of Personal Care And Service Workers
    4340.: "PRS", # .PRS-Animal Trainers
    4350.: "PRS", # .PRS-Animal Caretakers
    4400.: "PRS", # .PRS-Gambling Services Workers
    4420.: "PRS", # .PRS-Ushers, Lobby Attendants, And Ticket Takers
    4435.: "PRS", # .PRS-Other Entertainment Attendants And Related Workers
    4461.: "PRS", # .PRS-Embalmers, Crematory Operators, And Funeral Attendants
    4465.: "PRS", # .PRS-Morticians, Undertakers, And Funeral Arrangers
    4500.: "PRS", # .PRS-Barbers
    4510.: "PRS", # .PRS-Hairdressers, Hairstylists, And Cosmetologists
    4521.: "PRS", # .PRS-Manicurists And Pedicurists
    4522.: "PRS", # .PRS-Skincare Specialists
    4525.: "PRS", # .PRS-Other Personal Appearance Workers
    4530.: "PRS", # .PRS-Baggage Porters, Bellhops, And Concierges
    4540.: "PRS", # .PRS-Tour And Travel Guides
    4600.: "PRS", # .PRS-Childcare Workers
    4621.: "PRS", # .PRS-Exercise Trainers And Group Fitness Instructors
    4622.: "PRS", # .PRS-Recreation Workers
    4640.: "PRS", # .PRS-Residential Advisors
    4655.: "PRS", # .PRS-Personal Care and Service Workers, All Other
    4700.: "SAL", # .SAL-First-Line Supervisors Of Retail Sales Workers
    4710.: "SAL", # .SAL-First-Line Supervisors Of Non-Retail Sales Workers
    4720.: "SAL", # .SAL-Cashiers
    4740.: "SAL", # .SAL-Counter And Rental Clerks
    4750.: "SAL", # .SAL-Parts Salespersons
    4760.: "SAL", # .SAL-Retail Salespersons
    4800.: "SAL", # .SAL-Advertising Sales Agents
    4810.: "SAL", # .SAL-Insurance Sales Agents
    4820.: "SAL", # .SAL-Securities, Commodities, And Financial Services Sales .Agents
    4830.: "SAL", # .SAL-Travel Agents
    4840.: "SAL", # .SAL-Sales Representatives Of Services, Except Advertising, .Insurance, Financial Services, And Travel
    4850.: "SAL", # .SAL-Sales Representatives, Wholesale And Manufacturing
    4900.: "SAL", # .SAL-Models, Demonstrators, And Product Promoters
    4920.: "SAL", # .SAL-Real Estate Brokers And Sales Agents
    4930.: "SAL", # .SAL-Sales Engineers
    4940.: "SAL", # .SAL-Telemarketers
    4950.: "SAL", # .SAL-Door-To-Door Sales Workers, News And Street Vendors, .And Related Workers
    4965.: "SAL", # .SAL-Sales And Related Workers, All Other
    5000.: "OFF", # .OFF-First-Line Supervisors Of Office And Administrative .Support Workers
    5010.: "OFF", # .OFF-Switchboard Operators, Including Answering Service
    5020.: "OFF", # .OFF-Telephone Operators
    5040.: "OFF", # .OFF-Communications Equipment Operators, All Other
    5100.: "OFF", # .OFF-Bill And Account Collectors
    5110.: "OFF", # .OFF-Billing And Posting Clerks
    5120.: "OFF", # .OFF-Bookkeeping, Accounting, And Auditing Clerks
    5140.: "OFF", # .OFF-Payroll And Timekeeping Clerks
    5150.: "OFF", # .OFF-Procurement Clerks
    5160.: "OFF", # .OFF-Tellers
    5165.: "OFF", # .OFF-Other Financial Clerks
    5220.: "OFF", # .OFF-Court, Municipal, And License Clerks
    5230.: "OFF", # .OFF-Credit Authorizers, Checkers, And Clerks
    5240.: "OFF", # .OFF-Customer Service Representatives
    5250.: "OFF", # .OFF-Eligibility Interviewers, Government Programs
    5260.: "OFF", # .OFF-File Clerks
    5300.: "OFF", # .OFF-Hotel, Motel, And Resort Desk Clerks
    5310.: "OFF", # .OFF-Interviewers, Except Eligibility And Loan
    5320.: "OFF", # .OFF-Library Assistants, Clerical
    5330.: "OFF", # .OFF-Loan Interviewers And Clerks
    5340.: "OFF", # .OFF-New Accounts Clerks
    5350.: "OFF", # .OFF-Correspondence Clerks And Order Clerks
    5360.: "OFF", # .OFF-Human Resources Assistants, Except Payroll And .Timekeeping
    5400.: "OFF", # .OFF-Receptionists And Information Clerks
    5410.: "OFF", # .OFF-Reservation And Transportation Ticket Agents And Travel .Clerks
    5420.: "OFF", # .OFF-Other Information And Records Clerks
    5500.: "OFF", # .OFF-Cargo And Freight Agents
    5510.: "OFF", # .OFF-Couriers And Messengers
    5521.: "OFF", # .OFF-Public Safety Telecommunicators
    5522.: "OFF", # .OFF-Dispatchers, Except Police, Fire, And Ambulance
    5530.: "OFF", # .OFF-Meter Readers, Utilities
    5540.: "OFF", # .OFF-Postal Service Clerks
    5550.: "OFF", # .OFF-Postal Service Mail Carriers
    5560.: "OFF", # .OFF-Postal Service Mail Sorters, Processors, And Processing .Machine Operators
    5600.: "OFF", # .OFF-Production, Planning, And Expediting Clerks
    5610.: "OFF", # .OFF-Shipping, Receiving, And Inventory Clerks
    5630.: "OFF", # .OFF-Weighers, Measurers, Checkers, And Samplers, .Recordkeeping
    5710.: "OFF", # .OFF-Executive Secretaries And Executive Administrative .Assistants
    5720.: "OFF", # .OFF-Legal Secretaries and Administrative Assistants
    5730.: "OFF", # .OFF-Medical Secretaries and Administrative Assistants
    5740.: "OFF", # .OFF-Secretaries And Administrative Assistants, Except .Legal, Medial, And Executive
    5810.: "OFF", # .OFF-Data Entry Keyers
    5820.: "OFF", # .OFF-Word Processors And Typists
    5840.: "OFF", # .OFF-Insurance Claims And Policy Processing Clerks
    5850.: "OFF", # .OFF-Mail Clerks And Mail Machine Operators, Except Postal .Service
    5860.: "OFF", # .OFF-Office Clerks, General
    5900.: "OFF", # .OFF-Office Machine Operators, Except Computer
    5910.: "OFF", # .OFF-Proofreaders And Copy Markers
    5920.: "OFF", # .OFF-Statistical Assistants
    5940.: "OFF", # .OFF-Other Office And Administrative Support Workers
    6005.: "FFF", # .FFF-First-Line Supervisors Of Farming, Fishing, And .Forestry Workers
    6010.: "FFF", # .FFF-Agricultural Inspectors
    6040.: "FFF", # .FFF-Graders And Sorters, Agricultural Products
    6050.: "FFF", # .FFF-Other Agricultural Workers
    6115.: "FFF", # .FFF-Fishing And Hunting Workers
    6120.: "FFF", # .FFF-Forest And Conservation Workers
    6130.: "FFF", # .FFF-Logging Workers
    6200.: "CON", # .CON-First-Line Supervisors Of Construction Trades And .Extraction Workers
    6210.: "CON", # .CON-Boilermakers
    6220.: "CON", # .CON-Brickmasons, Blockmasons, Stonemasons, And Reinforcing .Iron And Rebar Workers
    6230.: "CON", # .CON-Carpenters
    6240.: "CON", # .CON-Carpet, Floor, And Tile Installers And Finishers
    6250.: "CON", # .CON-Cement Masons, Concrete Finishers, And Terrazzo Workers
    6260.: "CON", # .CON-Construction Laborers
    6305.: "CON", # .CON-Construction Equipment Operators
    6330.: "CON", # .CON-Drywall Installers, Ceiling Tile Installers, And Tapers
    6355.: "CON", # .CON-Electricians
    6360.: "CON", # .CON-Glaziers
    6400.: "CON", # .CON-Insulation Workers
    6410.: "CON", # .CON-Painters and Paperhangers
    6441.: "CON", # .CON-Pipelayers
    6442.: "CON", # .CON-Plumbers, Pipefitters, And Steamfitters
    6460.: "CON", # .CON-Plasterers And Stucco Masons
    6515.: "CON", # .CON-Roofers
    6520.: "CON", # .CON-Sheet Metal Workers
    6530.: "CON", # .CON-Structural Iron And Steel Workers
    6540.: "CON", # .CON-Solar Photovoltaic Installers
    6600.: "CON", # .CON-Helpers, Construction Trades
    6660.: "CON", # .CON-Construction And Building Inspectors
    6700.: "CON", # .CON-Elevator Installers And Repairers
    6710.: "CON", # .CON-Fence Erectors
    6720.: "CON", # .CON-Hazardous Materials Removal Workers
    6730.: "CON", # .CON-Highway Maintenance Workers
    6740.: "CON", # .CON-Rail-Track Laying And Maintenance Equipment Operators
    6765.: "CON", # .CON-Other Construction And Related Workers
    6800.: "EXT", # .EXT-Derrick, Rotary Drill, And Service Unit Operators, And .Roustabouts, Oil, Gas, And Mining
    6825.: "EXT", # .EXT-Surface Mining Machine Operators And Earth Drillers
    6835.: "EXT", # .EXT-Explosives Workers, Ordnance Handling Experts, and .Blasters
    6850.: "EXT", # .EXT-Underground Mining Machine Operators
    6950.: "EXT", # .EXT-Other Extraction Workers
    7000.: "RPR", # .RPR-First-Line Supervisors Of Mechanics, Installers, And .Repairers
    7010.: "RPR", # .RPR-Computer, Automated Teller, And Office Machine .Repairers
    7020.: "RPR", # .RPR-Radio And Telecommunications Equipment Installers And .Repairers
    7030.: "RPR", # .RPR-Avionics Technicians
    7040.: "RPR", # .RPR-Electric Motor, Power Tool, And Related Repairers
    7100.: "RPR", # .RPR-Other Electrical And Electronic Equipment Mechanics, .Installers, And Repairers.
    7120.: "RPR", # .RPR-Electronic Home Entertainment Equipment Installers And .Repairers
    7130.: "RPR", # .RPR-Security And Fire Alarm Systems Installers
    7140.: "RPR", # .RPR-Aircraft Mechanics And Service Technicians
    7150.: "RPR", # .RPR-Automotive Body And Related Repairers
    7160.: "RPR", # .RPR-Automotive Glass Installers And Repairers
    7200.: "RPR", # .RPR-Automotive Service Technicians And Mechanics
    7210.: "RPR", # .RPR-Bus And Truck Mechanics And Diesel Engine Specialists
    7220.: "RPR", # .RPR-Heavy Vehicle And Mobile Equipment Service Technicians .And Mechanics
    7240.: "RPR", # .RPR-Small Engine Mechanics
    7260.: "RPR", # .RPR-Miscellaneous Vehicle And Mobile Equipment Mechanics, .Installers, And Repairers
    7300.: "RPR", # .RPR-Control And Valve Installers And Repairers
    7315.: "RPR", # .RPR-Heating, Air Conditioning, And Refrigeration Mechanics .And Installers
    7320.: "RPR", # .RPR-Home Appliance Repairers
    7330.: "RPR", # .RPR-Industrial And Refractory Machinery Mechanics
    7340.: "RPR", # .RPR-Maintenance And Repair Workers, General
    7350.: "RPR", # .RPR-Maintenance Workers, Machinery
    7360.: "RPR", # .RPR-Millwrights
    7410.: "RPR", # .RPR-Electrical Power-Line Installers And Repairers
    7420.: "RPR", # .RPR-Telecommunications Line Installers And Repairers
    7430.: "RPR", # .RPR-Precision Instrument And Equipment Repairers
    7510.: "RPR", # .RPR-Coin, Vending, And Amusement Machine Servicers And .Repairers
    7540.: "RPR", # .RPR-Locksmiths And Safe Repairers
    7560.: "RPR", # .RPR-Riggers
    7610.: "RPR", # .RPR-Helpers--Installation, Maintenance, And Repair Workers
    7640.: "RPR", # .RPR-Other Installation, Maintenance, And Repair Workers
    7700.: "PRD", # .PRD-First-Line Supervisors Of Production And Operating .Workers
    7720.: "PRD", # .PRD-Electrical, Electronics, And Electromechanical .Assemblers
    7730.: "PRD", # .PRD-Engine And Other Machine Assemblers
    7740.: "PRD", # .PRD-Structural Metal Fabricators And Fitters
    7750.: "PRD", # .PRD-Other Assemblers And Fabricators
    7800.: "PRD", # .PRD-Bakers
    7810.: "PRD", # .PRD-Butchers And Other Meat, Poultry, And Fish Processing .Workers
    7830.: "PRD", # .PRD-Food And Tobacco Roasting, Baking, And Drying Machine .Operators And Tenders
    7840.: "PRD", # .PRD-Food Batchmakers
    7850.: "PRD", # .PRD-Food Cooking Machine Operators And Tenders
    7855.: "PRD", # .PRD-Food Processing Workers, All Other
    7905.: "PRD", # .PRD-Computer Numerically Controlled Tool Operators And .Programmers
    7925.: "PRD", # .PRD-Forming Machine Setters, Operators, And Tenders, Metal .And Plastic
    7950.: "PRD", # .PRD-Cutting, Punching, And Press Machine Setters, .Operators, And Tenders, Metal And Plastic
    8000.: "Gri", # .Grinding, Lapping, Polishing, And Buffing Machine Tool
    8025.: "PRD", # .PRD-Other Machine Tool Setters, Operators, And Tenders, .Metal and Plastic
    8030.: "PRD", # .PRD-Machinists
    8040.: "PRD", # .PRD-Metal Furnace Operators, Tenders, Pourers, And Casters
    8100.: "PRD", # .PRD-Model Makers, Patternmakers, And Molding Machine .Setters, Metal And Plastic
    8130.: "PRD", # .PRD-Tool And Die Makers
    8140.: "PRD", # .PRD-Welding, Soldering, And Brazing Workers
    8225.: "PRD", # .PRD-Other Metal Workers And Plastic Workers
    8250.: "PRD", # .PRD-Prepress Technicians And Workers
    8255.: "PRD", # .PRD-Printing Press Operators
    8256.: "PRD", # .PRD-Print Binding And Finishing Workers
    8300.: "PRD", # .PRD-Laundry And Dry-Cleaning Workers
    8310.: "PRD", # .PRD-Pressers, Textile, Garment, And Related Materials
    8320.: "PRD", # .PRD-Sewing Machine Operators
    8335.: "PRD", # .PRD-Shoe And Leather Workers
    8350.: "PRD", # .PRD-Tailors, Dressmakers, And Sewers
    8365.: "PRD", # .PRD-Textile Machine Setters, Operators, And Tenders
    8410.: "PRD", # .PRD - Unknown (not in ACS PUMS data dictionary)
    8450.: "PRD", # .PRD-Upholsterers
    8465.: "PRD", # .PRD-Other Textile, Apparel, And Furnishings Workers
    8500.: "PRD", # .PRD-Cabinetmakers And Bench Carpenters
    8510.: "PRD", # .PRD-Furniture Finishers
    8530.: "PRD", # .PRD-Sawing Machine Setters, Operators, And Tenders, Wood
    8540.: "PRD", # .PRD-Woodworking Machine Setters, Operators, And Tenders, .Except Sawing
    8555.: "PRD", # .PRD-Other Woodworkers
    8600.: "PRD", # .PRD-Power Plant Operators, Distributors, And Dispatchers
    8610.: "PRD", # .PRD-Stationary Engineers And Boiler Operators
    8620.: "PRD", # .PRD-Water And Wastewater Treatment Plant And System .Operators
    8630.: "PRD", # .PRD-Miscellaneous Plant And System Operators
    8640.: "PRD", # .PRD-Chemical Processing Machine Setters, Operators, And .Tenders
    8650.: "PRD", # .PRD-Crushing, Grinding, Polishing, Mixing, And Blending .Workers
    8710.: "PRD", # .PRD-Cutting Workers
    8720.: "PRD", # .PRD-Extruding, Forming, Pressing, And Compacting Machine .Setters, Operators, And Tenders
    8730.: "PRD", # .PRD-Furnace, Kiln, Oven, Drier, And Kettle Operators And .Tenders
    8740.: "PRD", # .PRD-Inspectors, Testers, Sorters, Samplers, And Weighers
    8750.: "PRD", # .PRD-Jewelers And Precious Stone And Metal Workers
    8760.: "PRD", # .PRD-Dental And Ophthalmic Laboratory Technicians And .Medical Appliance Technicians
    8800.: "PRD", # .PRD-Packaging And Filling Machine Operators And Tenders
    8810.: "PRD", # .PRD-Painting Workers
    8830.: "PRD", # .PRD-Photographic Process Workers And Processing Machine .Operators
    8850.: "PRD", # .PRD-Adhesive Bonding Machine Operators And Tenders
    8910.: "PRD", # .PRD-Etchers And Engravers
    8920.: "PRD", # .PRD-Molders, Shapers, And Casters, Except Metal And Plastic
    8930.: "PRD", # .PRD-Paper Goods Machine Setters, Operators, And Tenders
    8940.: "PRD", # .PRD-Tire Builders
    8950.: "PRD", # .PRD-Helpers-Production Workers
    8990.: "PRD", # .PRD-Miscellaneous Production Workers, Including Equipment .Operators And Tenders
    9005.: "TRN", # .TRN-Supervisors Of Transportation And Material Moving .Workers
    9030.: "TRN", # .TRN-Aircraft Pilots And Flight Engineers
    9040.: "TRN", # .TRN-Air Traffic Controllers And Airfield Operations .Specialists
    9050.: "TRN", # .TRN-Flight Attendants
    9110.: "TRN", # .TRN-Ambulance Drivers And Attendants, Except Emergency .Medical Technicians
    9121.: "TRN", # .TRN-Bus Drivers, School
    9122.: "TRN", # .TRN-Bus Drivers, Transit And Intercity
    9130.: "TRN", # .TRN-Driver/Sales Workers And Truck Drivers
    9141.: "TRN", # .TRN-Shuttle Drivers And Chauffeurs
    9142.: "TRN", # .TRN-Taxi Drivers
    9150.: "TRN", # .TRN-Motor Vehicle Operators, All Other
    9210.: "TRN", # .TRN-Locomotive Engineers And Operators
    9240.: "TRN", # .TRN-Railroad Conductors And Yardmasters
    9265.: "TRN", # .TRN-Other Rail Transportation Workers
    9300.: "TRN", # .TRN-Sailors And Marine Oilers, And Ship Engineers
    9310.: "TRN", # .TRN-Ship And Boat Captains And Operators
    9350.: "TRN", # .TRN-Parking Lot Attendants
    9365.: "TRN", # .TRN-Transportation Service Attendants
    9410.: "TRN", # .TRN-Transportation Inspectors
    9415.: "TRN", # .TRN-Passenger Attendants
    9430.: "TRN", # .TRN-Other Transportation Workers
    9510.: "TRN", # .TRN-Crane And Tower Operators
    9570.: "TRN", # .TRN-Conveyor, Dredge, And Hoist and Winch Operators
    9600.: "TRN", # .TRN-Industrial Truck And Tractor Operators
    9610.: "TRN", # .TRN-Cleaners Of Vehicles And Equipment
    9620.: "TRN", # .TRN-Laborers And Freight, Stock, And Material Movers, Hand
    9630.: "TRN", # .TRN-Machine Feeders And Offbearers
    9640.: "TRN", # .TRN-Packers And Packagers, Hand
    9645.: "TRN", # .TRN-Stockers And Order Fillers
    9650.: "TRN", # .TRN-Pumping Station Operators
    9720.: "TRN", # .TRN-Refuse And Recyclable Material Collectors
    9760.: "TRN", # .TRN-Other Material Moving Workers
    9800.: "MIL", # .MIL-Military Officer Special And Tactical Operations .Leaders
    9810.: "MIL", # .MIL-First-Line Enlisted Military Supervisors
    9825.: "MIL", # .MIL-Military Enlisted Tactical Operations And Air/Weapons .Specialists And Crew Members
    9830.: "MIL", # .MIL-Military, Rank Not Specified
    9920.: "Une", # .Unemployed And Last Worked 5 Years Ago Or Earlier Or Never .Worked
    1020.: "UNK",  # Unknown category not documented in ACS PUMS documentation.
}

OCCP_MAPPING_IDENTITY = {
    k:str(k) for k,_ in OCCP_MAPPING_COARSE.items()
}

def _float_to_string_mapping(minval, maxval):
    return {float(x): f"{x:02d}" for x in range(minval, maxval+1)}


DEFAULT_ACS_FEATURE_MAPPINGS = {
    # Citizenship
    'CIT': _float_to_string_mapping(1, 5),
    # "Class of worker"
    'COW': _float_to_string_mapping(1, 9),
    # Division code based on 2010 Census
    'DIVISION': _float_to_string_mapping(0, 9),
    # Ability to speak English
    'ENG': _float_to_string_mapping(0, 4),
    # Gave birth to child within the past 12 months
    'FER': _float_to_string_mapping(0, 2),
    # Insurance through a current or former employer or union
    'HINS1': _float_to_string_mapping(1, 2),
    # Insurance purchased directly from an insurance company
    'HINS2': _float_to_string_mapping(1, 2),
    # Medicare, for people 65 and older, or people with certain disabilities
    'HINS3': _float_to_string_mapping(1, 2),
    # Medicaid, Medical Assistance, or any kind of government-assistance plan for those with low incomes or a disability
    'HINS4': _float_to_string_mapping(1, 2),
    # Marital status.
    'MAR': _float_to_string_mapping(0, 5),
    # On layoff from work (Unedited-See "Employment Status Recode" (ESR))
    'NWLA': _float_to_string_mapping(1, 3),
    # Looking for work (Unedited-See "Employment Status Recode" (ESR))
    'NWLK': _float_to_string_mapping(1, 3),
    # # Occupation recode for 2018 and later based on 2018 OCC codes.
    # 'OCCP': OCCP_MAPPING_FINE,
    # Place of birth.
    'POBP': POBP_MAPPING,
    # Relationship
    'RELP': _float_to_string_mapping(0, 17),
    # Educational attainment
    'SCHL': _float_to_string_mapping(1, 24),
    # Worked last week
    'WRK': _float_to_string_mapping(0, 2)
}

OCCP_MAPPINGS = {
    'identity': OCCP_MAPPING_IDENTITY,
    'coarse': OCCP_MAPPING_COARSE,
    'fine': OCCP_MAPPING_FINE,
}


def get_feature_mapping(occp_mapping='identity') -> frozendict:
    """Helper function to fetch feature mapping dict.

    Returns a nested dict mapping feature names to a mapping;
    the mapping assigns each possible value of the feature
    to a new set of values (in most cases this is either
    a 1:1 mapping or a many:1 mapping to reduce cardinality).
    """
    assert occp_mapping in ('identity', 'coarse', 'fine')
    mapping = DEFAULT_ACS_FEATURE_MAPPINGS
    mapping['OCCP'] = OCCP_MAPPINGS[occp_mapping]
    return frozendict(mapping)
