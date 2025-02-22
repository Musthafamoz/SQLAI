PGDMP                      |            denver    16.2    16.1 *    ^           0    0    ENCODING    ENCODING        SET client_encoding = 'UTF8';
                      false            _           0    0 
   STDSTRINGS 
   STDSTRINGS     (   SET standard_conforming_strings = 'on';
                      false            `           0    0 
   SEARCHPATH 
   SEARCHPATH     8   SELECT pg_catalog.set_config('search_path', '', false);
                      false            a           1262    31633    denver    DATABASE     h   CREATE DATABASE denver WITH TEMPLATE = template0 ENCODING = 'UTF8' LOCALE_PROVIDER = libc LOCALE = 'C';
    DROP DATABASE denver;
                postgres    false            �            1259    32078    AllKD-Monthly    TABLE     �  CREATE TABLE public."AllKD-Monthly" (
    "Current Year" character varying(255),
    "Current Year Forecast" character varying(255),
    "Current Year Budget" character varying(255),
    "Const. & Field Svcs-SOS" character varying(255),
    "Budget-SOS" character varying(255),
    "Utilities-SOS" character varying(255),
    "Budget-SOS_1" character varying(255),
    "Current Year Forecast SOS" character varying(255),
    "Current Year Budget SOS" character varying(255),
    "Waste disposal-WQT" character varying(255),
    "Budget-WQT" character varying(255),
    "Utilities-WQT" character varying(255),
    "Budget-WQT_2" character varying(255),
    "Current Year Forecast WQT" character varying(255),
    "Current Year Budget WQT" character varying(255),
    "Chemicals-WQT" character varying(255),
    "Budget" character varying(255),
    "Const Field svcs w/o Paving, Hauling" character varying(255),
    "Budget_3" character varying(255),
    "Paving" character varying(255),
    "Budget_4" character varying(255),
    "Hauling/Trucking" character varying(255),
    "Budget_5" character varying(255),
    "Waste Disposal" character varying(255),
    "Budget_6" character varying(255),
    "Current Year Forecast WD" character varying(255),
    "Current Year Budget WD" character varying(255),
    "Facility Services" character varying(255),
    "Budget_7" character varying(255),
    "Utilities" character varying(255),
    "Budget_8" character varying(255),
    "Mat'l Purch/Issued" character varying(255),
    "Budget_9" character varying(255),
    "Fuel" character varying(255),
    "Budget_10" character varying(255),
    "Inventory adj'ts" character varying(255),
    "Prior Year" character varying(255),
    "Prior Year Actuals" character varying(255),
    "Const. & Field Svcs-CSF" character varying(255),
    "Utilities_13" character varying(255),
    "Prior Year Actuals SOS" character varying(255),
    "Waste disposal-WQT_14" character varying(255),
    "Utilities-WQT_15" character varying(255),
    "Prior Year Actuals WQT" character varying(255),
    "Chemicals-WQT_17" character varying(255),
    "Const Field svcs w/o Paving, Hauling_18" character varying(255),
    "Paving_19" character varying(255),
    "Hauling/Trucking_20" character varying(255),
    "Waste Disposal_21" character varying(255),
    "Prior Year Actual WD" character varying(255),
    "Facility Services_22" character varying(255),
    "Utilities_23" character varying(255),
    "Mat'l Purch/Issued_24" character varying(255),
    "Fuel_25" character varying(255),
    "Inventory adj'ts_26" character varying(255),
    "Inventory adj'ts_27" character varying(255),
    "Current Year-ALLKD" character varying(255),
    "Current Year Forecast Cumulative" character varying(255),
    "Current Year Budget Cumulative" character varying(255),
    "Const. & Field Svcs" character varying(255),
    "Budget_28" character varying(255),
    "Utilities_29" character varying(255),
    "Budget_30" character varying(255),
    "Current Year Forecast SOS Cumulative" character varying(255),
    "Current Year Budget SOS Cumulative" character varying(255),
    "Prof. services" character varying(255),
    "Budget_31" character varying(255),
    "Waste disposal-PWQT" character varying(255),
    "Budget-PWQT" character varying(255),
    "Utilities-PWQT" character varying(255),
    "Budget-PWQT_32" character varying(255),
    "Current Year Forecast WQT Cumulative" character varying(255),
    "Current Year Budget WQT Cumulative" character varying(255),
    "Chemicals-PWQT" character varying(255),
    "Budget-PWQT_33" character varying(255),
    "Const Field svcs w/o Paving, Hauling_34" character varying(255),
    "Budget_35" character varying(255),
    "Paving_36" character varying(255),
    "Budget_37" character varying(255),
    "Hauling/Trucking_38" character varying(255),
    "Budget_39" character varying(255),
    "Waste Disposal_40" character varying(255),
    "Budget_41" character varying(255),
    "Current Year Forecast WD Cumulative" character varying(255),
    "Current Year Budget WD Cumulative" character varying(255),
    "Facility Services_42" character varying(255),
    "Budget_43" character varying(255),
    "Utilities_44" character varying(255),
    "Budget_45" character varying(255),
    "Mat'l Purch/Issued_46" character varying(255),
    "Budget_47" character varying(255),
    "Fuel_48" character varying(255),
    "Budget_49" character varying(255),
    "Inventory adj'ts_50" character varying(255),
    "Previous Year-AKD" character varying(255),
    "Prior Year Actual Cumulative" character varying(255),
    "Const. & Field Svcs_53" character varying(255),
    "Utilities_54" character varying(255),
    "Prior Year Actuals SOS Cumulative" character varying(255),
    "Prof. services_55" character varying(255),
    "Waste disposal.1" character varying(255),
    "Utilities_56" character varying(255),
    "Prior Year Actuals WQT Cumulative" character varying(255),
    "Chemicals" character varying(255),
    "Const Field svcs w/o Paving, Hauling_57" character varying(255),
    "Paving_58" character varying(255),
    "Hauling/Trucking_59" character varying(255),
    "Waste Disposal_60" character varying(255),
    "Prior Year Actual WD Cumulative" character varying(255),
    "Facility Services_61" character varying(255),
    "Utilities_62" character varying(255),
    "Mat'l Purch/Issued_63" character varying(255),
    "Fuel_64" character varying(255),
    "Inventory adj'ts_65" character varying(255),
    "_Month" character varying(255),
    "MonthNum" character varying(255)
);
 #   DROP TABLE public."AllKD-Monthly";
       public         heap    postgres    false            �            1259    31665    BU_NAME_(Section_All)    TABLE     l  CREATE TABLE public."BU_NAME_(Section_All)" (
    "Exp-Type" text,
    "Exp-SubType" text,
    "Current Budget" character varying,
    "YTD Actuals" character varying,
    "Rem. Mo.Forecast" character varying,
    "Full Year Forecast" character varying,
    "Current Mo.Full Year Forecast" character varying,
    "Prior Mo.Full Year Forecast" character varying
);
 +   DROP TABLE public."BU_NAME_(Section_All)";
       public         heap    postgres    false            �            1259    31676 
   Cumulative    TABLE     �  CREATE TABLE public."Cumulative" (
    "Ledger_name" text,
    "1/1/2023" character varying(20),
    "1/2/2023" character varying(20),
    "1/3/2023" character varying(20),
    "1/4/2023" character varying(20),
    "1/5/2023" character varying(20),
    "1/6/2023" character varying(20),
    "1/7/2023" character varying(20),
    "1/8/2023" character varying(20),
    "1/9/2023" character varying(20),
    "1/10/2023" character varying(20),
    "1/11/2023" character varying(20),
    "1/12/2023" character varying(20),
    "1/1/2022" character varying(20),
    "1/2/2022" character varying(20),
    "1/3/2022" character varying(20),
    "1/4/2022" character varying(20),
    "1/5/2022" character varying(20),
    "1/6/2022" character varying(20),
    "1/7/2022" character varying(20),
    "1/8/2022" character varying(20),
    "1/9/2022" character varying(20),
    "1/10/2022" character varying(20),
    "1/11/2022" character varying(20),
    "1/12/2022" character varying(20)
);
     DROP TABLE public."Cumulative";
       public         heap    postgres    false            �            1259    31681    Cumulative Year 2    TABLE       CREATE TABLE public."Cumulative Year 2" (
    "Current Year" text,
    "1/1/2023" character varying(20),
    "1/2/2023" character varying(20),
    "1/3/2023" character varying(20),
    "1/4/2023" character varying(20),
    "1/5/2023" character varying(20),
    "1/6/2023" character varying(20),
    "1/7/2023" character varying(20),
    "1/8/2023" character varying(20),
    "1/9/2023" character varying(20),
    "1/10/2023" character varying(20),
    "1/11/2023" character varying(20),
    "1/12/2023" character varying(20)
);
 '   DROP TABLE public."Cumulative Year 2";
       public         heap    postgres    false            �            1259    31686    CumulativeTrans_SEC    TABLE       CREATE TABLE public."CumulativeTrans_SEC" (
    "Current Year" character varying(20),
    "Chief" character varying(20),
    "Budget" character varying(20),
    "CSF" character varying(20),
    "Budget_1" character varying(20),
    "SOS" character varying(20),
    "Budget_2" character varying(20),
    "WQT" character varying(20),
    "Budget_3" character varying(20),
    "WD" character varying(20),
    "Budget_4" character varying(20),
    "SSV" character varying(20),
    "Budget_5" character varying(20)
);
 )   DROP TABLE public."CumulativeTrans_SEC";
       public         heap    postgres    false            �            1259    31689    CumulativeTrans_Year    TABLE     N  CREATE TABLE public."CumulativeTrans_Year" (
    "Current Year" character varying(20),
    "Current Year Forecast" character varying(30),
    "Current Year Budget" character varying(30),
    "Prior Year Actuals" character varying(30),
    "Prior Year" character varying(20),
    "_Month" text,
    "_MnthNum" character varying(20)
);
 *   DROP TABLE public."CumulativeTrans_Year";
       public         heap    postgres    false            �            1259    31699    Display_Units    TABLE     n   CREATE TABLE public."Display_Units" (
    "DispUnit" text,
    "S.No" character varying,
    "S.name" text
);
 #   DROP TABLE public."Display_Units";
       public         heap    postgres    false            �            1259    31709    FP    TABLE     o  CREATE TABLE public."FP" (
    "EXP-Type" text,
    "Current Budget" character varying(50),
    "YTD Actuals" character varying(20),
    "Rem. Mo.Forecast" character varying(20),
    "Full Year Forecast" character varying(20),
    "Budget over/(under)" character varying(50),
    "Budget Variance %" character varying(50),
    "Report Order" character varying(20)
);
    DROP TABLE public."FP";
       public         heap    postgres    false            �            1259    31704    Financial Report    TABLE     �  CREATE TABLE public."Financial Report" (
    "EXP-Type" text,
    "Current Budget" character varying,
    "YTD Actuals" character varying,
    "Rem. Mo.Forecast" character varying,
    "Full Year Forecast" character varying,
    "Current Mo.Full Year Forecast" character varying,
    "Prior Mo.Full Year Forecast" character varying,
    "EXP_name" text,
    "Budget over/(under)" character varying,
    "Budget Variance %" character varying,
    "Report Order" character varying
);
 &   DROP TABLE public."Financial Report";
       public         heap    postgres    false            �            1259    31714    KD-ALL    TABLE     �  CREATE TABLE public."KD-ALL" (
    "Current Year" text,
    "Jan" character varying(30),
    "Feb" character varying(30),
    "Mar" character varying(30),
    "Apr" character varying(30),
    "May" character varying(20),
    "Jun" character varying(20),
    "July" character varying(20),
    "Aug" character varying(20),
    "Sept" character varying(20),
    "Oct" character varying(20),
    "Nov" character varying(20),
    "Dec" character varying(20),
    "BU_SUBNAME" text
);
    DROP TABLE public."KD-ALL";
       public         heap    postgres    false            �            1259    31719    MonthTB    TABLE     �   CREATE TABLE public."MonthTB" (
    "Month" character varying(30),
    "_Month" character varying(30),
    "_MonthNum" character varying(30)
);
    DROP TABLE public."MonthTB";
       public         heap    postgres    false            �            1259    31722    OM Change Over Time    TABLE     �  CREATE TABLE public."OM Change Over Time" (
    "Current Year" character varying(20),
    "YE Forecast at Quarter End - Revenue" character varying(50),
    "YE Forecast at Quarter End Salaries & Benefits" character varying(50),
    "YE Forecast at Quarter End Professional Services/Contingent" character varying(100),
    "YE Forecast at Quarter End Purchased Services" character varying(50),
    "YE Forecast at Quarter End Materials, Supplies, Chemicals" character varying(100),
    "YE Forecast at Quarter End Other (not incl. Debt Service)" character varying(100),
    "YE Forecast at Quarter End(Excl S&B)" character varying(50),
    "YE Forecast at Quarter End (Incl S&B)" character varying(50),
    "2023 Budget - Revenue" character varying(50),
    "2023 Budget - Salaries & Benefits" character varying(50),
    "2023 Budget - Professional Services/Contingent" character varying(50),
    "2023 Budget - Purchased Services" character varying(50),
    "2023 Budget - Materials, Supplies, Chemicals" character varying(50),
    "2023 Budget - Other (not incl. Debt Service)" character varying(50),
    "2023 Budget (Excl S&B)" character varying(50),
    "2023 Budget(Incl S&B)" character varying(50),
    "Prior Year - Revenue" character varying(50),
    "Prior Year - Salaries & Benefits" character varying(50),
    "Prior Year - Professional Services/Contingent" character varying(50),
    "Prior Year - Purchased Services" character varying(50),
    "Prior Year - Materials, Supplies, Chemicals" character varying(50),
    "Prior Year - Other (not incl. Debt Service)" character varying(50),
    "Prior Year(Excl S&B)" character varying(50),
    "Prior Year(Incl S&B)" character varying(50)
);
 )   DROP TABLE public."OM Change Over Time";
       public         heap    postgres    false            �            1259    31732    Section_All    TABLE     �  CREATE TABLE public."Section_All" (
    "BU NAME" text,
    "Exp-Type" character varying(50),
    "Exp-SubType" character varying(50),
    "Current Budget" character varying(50),
    "YTD Actuals" character varying(50),
    "Rem. Mo.Forecast" character varying(50),
    "Full Year Forecast" character varying(50),
    "Current Mo.Full Year Forecast" character varying(50),
    "Prior Mo.Full Year Forecast" character varying(50)
);
 !   DROP TABLE public."Section_All";
       public         heap    postgres    false            �            1259    31738    Switch fields    TABLE     �   CREATE TABLE public."Switch fields" (
    "Switch fields" text,
    "Switch fields Fields" character varying(50),
    "Switch fields Order" character varying(20)
);
 #   DROP TABLE public."Switch fields";
       public         heap    postgres    false            �            1259    31748    TOTAL O&M-BU Summary    TABLE       CREATE TABLE public."TOTAL O&M-BU Summary" (
    "BU Name" character varying(100),
    "BU" integer,
    "Business Unit Name" character varying(100),
    "MW-Budget" character varying(20),
    "MW-Actual" character varying(20),
    "YTD-Budget" character varying(20),
    "YTD-Actual" character varying(20),
    "FYF-CurrentBudget" character varying(20),
    "FYF-YTDActuals" character varying(20),
    "JUL-Forecast" character varying(20),
    "AUG-Forecast" character varying(20),
    "SEP-Forecast" character varying(20),
    "OCT-Forecast" character varying(20),
    "NOV-Forecast" character varying(20),
    "DEC-Forecast" character varying(20),
    "FYF-YTD_REM-MO-Forcast" character varying(20),
    "FULL YEAR FORECAST" character varying(20),
    "BUSHORT" text
);
 *   DROP TABLE public."TOTAL O&M-BU Summary";
       public         heap    postgres    false            �            1259    31743    Table    TABLE     A  CREATE TABLE public."Table" (
    "S.No" integer,
    "$ in thousands" text,
    "CurrentBudget" character varying(20),
    "YTDActuals" character varying(20),
    "Rem. Mo.Forecast" character varying(20),
    "Full YearForecast" character varying(20),
    "BudgetOver/(Under)" character varying(20),
    "Budget Variance %" character varying(20),
    "Current Month Full Year Forecast" character varying(20),
    "Prior Month Full Year Forecast" character varying(20),
    "Prior Mo.Forecast" character varying(20),
    "AC NO" integer,
    "Account" character varying(60)
);
    DROP TABLE public."Table";
       public         heap    postgres    false            �            1259    31753 	   Wages_All    TABLE     G  CREATE TABLE public."Wages_All" (
    "BU NAME" character varying(100),
    "BU" integer,
    "Business Unit Name" character varying(100),
    "Ledger Name" character varying(100),
    "JAN-Budget" character varying(20),
    "FEB-Budget" character varying(20),
    "MAR-Budget" character varying(20),
    "APR-Budget" character varying(20),
    "MAY-Budget" character varying(20),
    "JUN-Budget" character varying(20),
    "JUL-Budget" character varying(20),
    "AUG-Budget" character varying(20),
    "SEP-Budget" character varying(20),
    "OCT-Budget" character varying(20),
    "NOV-Budget" character varying(20),
    "DEC-Budget" character varying(20),
    "JAN-Actual" character varying(20),
    "FEB-Actual" character varying(20),
    "MAR-Actual" character varying(20),
    "APR-Actual" character varying(20),
    "MAY-Actual" character varying(20),
    "JUN-Actual" character varying(20),
    "JUL-Forecast" character varying(20),
    "AUG-Forecast" character varying(20),
    "SEP-Forecast" character varying(20),
    "OCT-Forecast" character varying(20),
    "NOV-Forecast" character varying(20),
    "DEC-Forecast" character varying(20),
    "CurrentBudget" character varying(20),
    "YTDActuals" character varying(20),
    "Rem. Mo.Forecast" character varying(20),
    "Full YearForecast" character varying(20),
    "S.No" integer
);
    DROP TABLE public."Wages_All";
       public         heap    postgres    false            �            1259    31759    YEARTAB    TABLE     7   CREATE TABLE public."YEARTAB" (
    "Value" integer
);
    DROP TABLE public."YEARTAB";
       public         heap    postgres    false            �            1259    31727    s&b    TABLE     [  CREATE TABLE public."s&b" (
    "S.No" integer,
    "Salary Type" text,
    "Current Budget" character varying(30),
    "YTD Actuals" character varying(50),
    "Rem. Mo.Forecast" character varying(50),
    "Full Year Forecast" character varying(50),
    "Budget Over/Under" character varying(50),
    "Budget Variance %" character varying(50)
);
    DROP TABLE public."s&b";
       public         heap    postgres    false            [          0    32078    AllKD-Monthly 
   TABLE DATA           �	  COPY public."AllKD-Monthly" ("Current Year", "Current Year Forecast", "Current Year Budget", "Const. & Field Svcs-SOS", "Budget-SOS", "Utilities-SOS", "Budget-SOS_1", "Current Year Forecast SOS", "Current Year Budget SOS", "Waste disposal-WQT", "Budget-WQT", "Utilities-WQT", "Budget-WQT_2", "Current Year Forecast WQT", "Current Year Budget WQT", "Chemicals-WQT", "Budget", "Const Field svcs w/o Paving, Hauling", "Budget_3", "Paving", "Budget_4", "Hauling/Trucking", "Budget_5", "Waste Disposal", "Budget_6", "Current Year Forecast WD", "Current Year Budget WD", "Facility Services", "Budget_7", "Utilities", "Budget_8", "Mat'l Purch/Issued", "Budget_9", "Fuel", "Budget_10", "Inventory adj'ts", "Prior Year", "Prior Year Actuals", "Const. & Field Svcs-CSF", "Utilities_13", "Prior Year Actuals SOS", "Waste disposal-WQT_14", "Utilities-WQT_15", "Prior Year Actuals WQT", "Chemicals-WQT_17", "Const Field svcs w/o Paving, Hauling_18", "Paving_19", "Hauling/Trucking_20", "Waste Disposal_21", "Prior Year Actual WD", "Facility Services_22", "Utilities_23", "Mat'l Purch/Issued_24", "Fuel_25", "Inventory adj'ts_26", "Inventory adj'ts_27", "Current Year-ALLKD", "Current Year Forecast Cumulative", "Current Year Budget Cumulative", "Const. & Field Svcs", "Budget_28", "Utilities_29", "Budget_30", "Current Year Forecast SOS Cumulative", "Current Year Budget SOS Cumulative", "Prof. services", "Budget_31", "Waste disposal-PWQT", "Budget-PWQT", "Utilities-PWQT", "Budget-PWQT_32", "Current Year Forecast WQT Cumulative", "Current Year Budget WQT Cumulative", "Chemicals-PWQT", "Budget-PWQT_33", "Const Field svcs w/o Paving, Hauling_34", "Budget_35", "Paving_36", "Budget_37", "Hauling/Trucking_38", "Budget_39", "Waste Disposal_40", "Budget_41", "Current Year Forecast WD Cumulative", "Current Year Budget WD Cumulative", "Facility Services_42", "Budget_43", "Utilities_44", "Budget_45", "Mat'l Purch/Issued_46", "Budget_47", "Fuel_48", "Budget_49", "Inventory adj'ts_50", "Previous Year-AKD", "Prior Year Actual Cumulative", "Const. & Field Svcs_53", "Utilities_54", "Prior Year Actuals SOS Cumulative", "Prof. services_55", "Waste disposal.1", "Utilities_56", "Prior Year Actuals WQT Cumulative", "Chemicals", "Const Field svcs w/o Paving, Hauling_57", "Paving_58", "Hauling/Trucking_59", "Waste Disposal_60", "Prior Year Actual WD Cumulative", "Facility Services_61", "Utilities_62", "Mat'l Purch/Issued_63", "Fuel_64", "Inventory adj'ts_65", "_Month", "MonthNum") FROM stdin;
    public          postgres    false    233   �q       I          0    31665    BU_NAME_(Section_All) 
   TABLE DATA           �   COPY public."BU_NAME_(Section_All)" ("Exp-Type", "Exp-SubType", "Current Budget", "YTD Actuals", "Rem. Mo.Forecast", "Full Year Forecast", "Current Mo.Full Year Forecast", "Prior Mo.Full Year Forecast") FROM stdin;
    public          postgres    false    215   �       J          0    31676 
   Cumulative 
   TABLE DATA           [  COPY public."Cumulative" ("Ledger_name", "1/1/2023", "1/2/2023", "1/3/2023", "1/4/2023", "1/5/2023", "1/6/2023", "1/7/2023", "1/8/2023", "1/9/2023", "1/10/2023", "1/11/2023", "1/12/2023", "1/1/2022", "1/2/2022", "1/3/2022", "1/4/2022", "1/5/2022", "1/6/2022", "1/7/2022", "1/8/2022", "1/9/2022", "1/10/2022", "1/11/2022", "1/12/2022") FROM stdin;
    public          postgres    false    216   ��       K          0    31681    Cumulative Year 2 
   TABLE DATA           �   COPY public."Cumulative Year 2" ("Current Year", "1/1/2023", "1/2/2023", "1/3/2023", "1/4/2023", "1/5/2023", "1/6/2023", "1/7/2023", "1/8/2023", "1/9/2023", "1/10/2023", "1/11/2023", "1/12/2023") FROM stdin;
    public          postgres    false    217   >�       L          0    31686    CumulativeTrans_SEC 
   TABLE DATA           �   COPY public."CumulativeTrans_SEC" ("Current Year", "Chief", "Budget", "CSF", "Budget_1", "SOS", "Budget_2", "WQT", "Budget_3", "WD", "Budget_4", "SSV", "Budget_5") FROM stdin;
    public          postgres    false    218   ��       M          0    31689    CumulativeTrans_Year 
   TABLE DATA           �   COPY public."CumulativeTrans_Year" ("Current Year", "Current Year Forecast", "Current Year Budget", "Prior Year Actuals", "Prior Year", "_Month", "_MnthNum") FROM stdin;
    public          postgres    false    219   ?�       N          0    31699    Display_Units 
   TABLE DATA           G   COPY public."Display_Units" ("DispUnit", "S.No", "S.name") FROM stdin;
    public          postgres    false    220   ؛       P          0    31709    FP 
   TABLE DATA           �   COPY public."FP" ("EXP-Type", "Current Budget", "YTD Actuals", "Rem. Mo.Forecast", "Full Year Forecast", "Budget over/(under)", "Budget Variance %", "Report Order") FROM stdin;
    public          postgres    false    222   �       O          0    31704    Financial Report 
   TABLE DATA             COPY public."Financial Report" ("EXP-Type", "Current Budget", "YTD Actuals", "Rem. Mo.Forecast", "Full Year Forecast", "Current Mo.Full Year Forecast", "Prior Mo.Full Year Forecast", "EXP_name", "Budget over/(under)", "Budget Variance %", "Report Order") FROM stdin;
    public          postgres    false    221   ��       Q          0    31714    KD-ALL 
   TABLE DATA           �   COPY public."KD-ALL" ("Current Year", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "July", "Aug", "Sept", "Oct", "Nov", "Dec", "BU_SUBNAME") FROM stdin;
    public          postgres    false    223   %�       R          0    31719    MonthTB 
   TABLE DATA           C   COPY public."MonthTB" ("Month", "_Month", "_MonthNum") FROM stdin;
    public          postgres    false    224   װ       S          0    31722    OM Change Over Time 
   TABLE DATA             COPY public."OM Change Over Time" ("Current Year", "YE Forecast at Quarter End - Revenue", "YE Forecast at Quarter End Salaries & Benefits", "YE Forecast at Quarter End Professional Services/Contingent", "YE Forecast at Quarter End Purchased Services", "YE Forecast at Quarter End Materials, Supplies, Chemicals", "YE Forecast at Quarter End Other (not incl. Debt Service)", "YE Forecast at Quarter End(Excl S&B)", "YE Forecast at Quarter End (Incl S&B)", "2023 Budget - Revenue", "2023 Budget - Salaries & Benefits", "2023 Budget - Professional Services/Contingent", "2023 Budget - Purchased Services", "2023 Budget - Materials, Supplies, Chemicals", "2023 Budget - Other (not incl. Debt Service)", "2023 Budget (Excl S&B)", "2023 Budget(Incl S&B)", "Prior Year - Revenue", "Prior Year - Salaries & Benefits", "Prior Year - Professional Services/Contingent", "Prior Year - Purchased Services", "Prior Year - Materials, Supplies, Chemicals", "Prior Year - Other (not incl. Debt Service)", "Prior Year(Excl S&B)", "Prior Year(Incl S&B)") FROM stdin;
    public          postgres    false    225   m�       U          0    31732    Section_All 
   TABLE DATA           �   COPY public."Section_All" ("BU NAME", "Exp-Type", "Exp-SubType", "Current Budget", "YTD Actuals", "Rem. Mo.Forecast", "Full Year Forecast", "Current Mo.Full Year Forecast", "Prior Mo.Full Year Forecast") FROM stdin;
    public          postgres    false    227   ��       V          0    31738    Switch fields 
   TABLE DATA           i   COPY public."Switch fields" ("Switch fields", "Switch fields Fields", "Switch fields Order") FROM stdin;
    public          postgres    false    228   ��       X          0    31748    TOTAL O&M-BU Summary 
   TABLE DATA           O  COPY public."TOTAL O&M-BU Summary" ("BU Name", "BU", "Business Unit Name", "MW-Budget", "MW-Actual", "YTD-Budget", "YTD-Actual", "FYF-CurrentBudget", "FYF-YTDActuals", "JUL-Forecast", "AUG-Forecast", "SEP-Forecast", "OCT-Forecast", "NOV-Forecast", "DEC-Forecast", "FYF-YTD_REM-MO-Forcast", "FULL YEAR FORECAST", "BUSHORT") FROM stdin;
    public          postgres    false    230   �       W          0    31743    Table 
   TABLE DATA             COPY public."Table" ("S.No", "$ in thousands", "CurrentBudget", "YTDActuals", "Rem. Mo.Forecast", "Full YearForecast", "BudgetOver/(Under)", "Budget Variance %", "Current Month Full Year Forecast", "Prior Month Full Year Forecast", "Prior Mo.Forecast", "AC NO", "Account") FROM stdin;
    public          postgres    false    229   ��       Y          0    31753 	   Wages_All 
   TABLE DATA             COPY public."Wages_All" ("BU NAME", "BU", "Business Unit Name", "Ledger Name", "JAN-Budget", "FEB-Budget", "MAR-Budget", "APR-Budget", "MAY-Budget", "JUN-Budget", "JUL-Budget", "AUG-Budget", "SEP-Budget", "OCT-Budget", "NOV-Budget", "DEC-Budget", "JAN-Actual", "FEB-Actual", "MAR-Actual", "APR-Actual", "MAY-Actual", "JUN-Actual", "JUL-Forecast", "AUG-Forecast", "SEP-Forecast", "OCT-Forecast", "NOV-Forecast", "DEC-Forecast", "CurrentBudget", "YTDActuals", "Rem. Mo.Forecast", "Full YearForecast", "S.No") FROM stdin;
    public          postgres    false    231   ��       Z          0    31759    YEARTAB 
   TABLE DATA           ,   COPY public."YEARTAB" ("Value") FROM stdin;
    public          postgres    false    232   q      T          0    31727    s&b 
   TABLE DATA           �   COPY public."s&b" ("S.No", "Salary Type", "Current Budget", "YTD Actuals", "Rem. Mo.Forecast", "Full Year Forecast", "Budget Over/Under", "Budget Variance %") FROM stdin;
    public          postgres    false    226   �      [      x�嚻�,�qE����f�b�&A� $�rF�@AG���֊��{Yd� ���̈{����J*�z��R������*s\�*�^���{��z�K��ݵ]9�5�u�vϴ���=�hW��utnW�����n��_�Q��������wމ���ѵ�]yx������2�ȉ�]�|�]�ң�~�y����s�u�r�����;nS�`gi���ʵ������ir�ܮ�~���ޯ�y�t�<�7�f��$N�su�����sԻO�5�MN<f�,���>�R��y^�ʾJj,`rQ����B�7���~���Z��d����� ���Q,�������5���5�������/7��	�]� �n�����*�&�>�b-g���v�]�5��.,1�z�9ֵ�/�l6��&A��'��2�M�f�5~�}ڜ.�m��z�[�J�l���F$�,|'�$R�5y��Ω�弌�@�+���������f��;�5�\��%���f�37o��F*)ߋ���k�
{1��w��ܣM�B�L"�B�P�>�$�&��!��ޟ�<�P�	͙P�U��J<7�Q���z$!�JP�,���^� ʤ��3;�k{Q��b�|'y�w�i-½�_)>�`�PIe��nBDsiH�E�WR�1�Y4��OB'Iy�a��S��s m{o�s ���H�xP�c�ݶ�`'�z}�DN�+���a�w��~������O\7��f�H�	lQ7|�%R:�@�Z6_��T��!���H��u��<��Sdv�eK�T�������/�Ys!=�1�� %�a�e�R�(r�)_~T]<<
9t��A�ރ:>�_�탞nj)E>(�g�b�!�:���3�e�7�^�8l�U!�,0W"Z,;���M�S��3���,�~_GQQ��Ƃ^�d�OT����f���`����y�����a��5��/�χ.= +�0j?�������muK���s8�բĨ$��� �EJ�r�J��xs7�=	ٵ'q�7�Zi��g���n<����AoMe�S	�Ú`����%J��^>�VٴA\�(����-�E�$I��,K�t�%*81��BZ�Ì\v5	�r�dp��@9� �Dx��NF���#R�Z%k�uR�-������|6���	`�)O��ʷ���wP12�Q��?Pƕ�h��hR� ��0�:�8��n��5�7B�N�Od�UM�U��"�ʆ��2�t�.�Y��i��N��/��Vķ�aGh�'�x�T�)c�M+3Ŀ�Q��:�a�Bs@��̿X�� 8���N�g٣zy/�^s���C������@'p�Q0e�#����pR�*�ɡ��J{��_����񦛉xa�Z����^7�7,��V���]��I 4z�R�^�� n�{����������uw[P�|e�eN�VA�S�Gf��$����\V�f8���f�B���	Sx��� ����AEHBϚ�> "Ͳ�F��u�J��pg�t�J��|�ChY0_b���M.��mQyE�Z� �4��� [x�S�0��}CYq9�,%{��,�V�{_=x�K�Y�R��0N�kIęJn��d��%bZ�M���_��ىu�*����.��
`�NZ�VM�@S�V�[��Lz���e�Ѕ�#
����i��Q��ڰ���4�Ψa�s���(4^� �:�[�`ʁ;,YP器�s�-�������W(%eӉCn�B[�p�0k�<�Cf'��>@�� 4�vt"6>ۏ$]�Ƣk256T_Tזּ���C(�I 8�)��(�X�+4�E�XKu��"!SX*H0�T��h3��C����4S���@����d&w��S��&��
g;#�R]&U��F}�y�� N�~�MwH畕��M�Y3sf%�%�n��́0�z!DLbE�Ma"�5��Y�΁h��p�c(Av�N��o[�^v������E,r��E�
��N�=Q�QzD/-�X��U��M:�ǡ�=�.т���j��E�ԁk�����Y�i븭���ϰ���|���%�����a����G��<�V��d7�p]����t�!k�08�C�Yǣncʣ��&\	'�I�il����F�}��U\�ewm㫜w}�/�`Ѥ��\����ل�:D'9 O˲�Kr�6���"�֡�l]�@R�0D����5�g���&����8���`rPw9_v8)L�oW��oQV���\�͑x 1���PԬm��
��1�u;�I:]�sIc>�)�M6����pER��lў@3�4����o�,��ɐY:G޸!f��vXz�Rwa-����	�^6,��D��(�*hY���s4-Q�*'���_��O6u��^዇գ?�[����)���w��v����a	U�l�Vȣs�1$`Ġ�'b��s�~irZ��<E�|ه����R8
�B����֯����FC��Tn��KɊ�Y�`+q��A�?��6��ړ�-�U��}2�h�Ų�Wz��v�8�&���P�x\H��	�Zb�5K5�]6���������H�EmÂԁ�d3k��,�]���I��{�v��挺�N%k��&T�T����Ù��@W])
�gCA;�D���R�����p�^�	5rX a鋏����b;�>Pl���(K���M(�.	�KI=v�?���<_�\b?�U"�3l5�0������r�ߡ�af@Lt|,���Lբ���d2�i����l}�C�� 	���)�1�9(�ъc'�f����e���cB9j9dAi*�-�D:����OR�3�#�pi�w�|:���׿���)�UK�b>���Vٖ�
��b���yK��z��r��v����cs�H���m�U���QM���X6�������t�c�w&�b�%���8Jt�x��$Hz��\��F���H!ί�<*�l`���ي�����%�3��uk�)�N�{2��m��K<���A�0�u�Qq#,���vd���DQ�޹Jv�D�=w�S�Ԡ���>콢���m:�N��0����A�X����r��-�3U�vV���>v��˲�t�ET41pp�i�%�Ҙ�Pr,y�?-iкV�=A_%��	V�Jz� c�t��%8��^3�w�e����q�>]ugX.�y��M�4��Y�5iL����?��9W{b+N4��Wu�fL'�?�v��[&�mWu���,��ܶm`��I�Qؓ��[��©_��z>�̎���D�̾/ޙ�����v���[n)\�R�����G1#Z��^�	�t��ۇ�[ԲM��#JB'=TK�l_D�t�]�1���0G�����.�8k|n+�%�Q8��Y0f�i����y��"�M��]�r'8��N��U�͔�R�'Eim�ЍA[�r�qPl��->9�����P�nvD� �m��E�5���L��`�3���e1�������2�zT��A�qL, ���q�T�����.M�S��u� ^{���aS�sk�z�b�y� q�^#��Xˬ����͸{�1�|K�s�F��i��(%2٨�?�.PS>*%�jvՁ?y����l�X�,��B�;pb����y0����[�=�c��CU�Q�C��7�ˇ�ڙ�x�f{��������w*����X�)��+����m��lD�M����SC�ڱh�k!.��J���S�8�9}t�R;�~M����V�g߄jȒ�֞cR��t��/��{hw����{�!�i7R�s�e�S��#��<J8U��3�E��3=.��G[�Z��>��3����w/���tE��BŶ���yV�����Wz졳;*�c�b��bv�LL��<�����.7U;;�)�WIw�&���a�l}ws�{J���\>	�L]/�f8!Jku�Em��W���B��[���y幝��BP�
�l��Mr�����z�8>��EP��-��yI��kj������M�!� �'�ꪃ�e;.ܖ=E��O�b��ғ���$L��[̢[���J)�P5�gP]�,��[5�Qά �  F�Y��_bM�������I3���o�Zn^�;뾶��ӻ�=�j����I�~��3�-1��Δ�c�yv���,���h��(=n�f��px�ݠڟZޅ(4��;7ͮc~vޅ��O��VP������s����lqD6��~�P�K�~�.�E�g�����T��ر��"]�3����S%4@vH���Ĝ$���,O7�iex_�S��ҧlLUg�ű,�>�[+$����Q�C_7�^��5c�1C��(n9ꮔ��"�����顑sG�ok�Fv6�z8�����+Gw�썐yH�P-��(���#E��IA�K{���q.!�[̎�&s\&)p��4R��c��`�v�7�َ���dp��̬B#u�vU��l�w��*�3��K�	M:;��^�3�P �w#�}-aXS3��j~v���HU���K��G�ĝ���I�jq���=��%7xR�V�h��=���R�}6���aV$=�#W�,s6�����L7�'���U1��a��":[��Su����O�1�q|i4�w��=
��#"�����t�n[������Ӊ.�[3V�\ZTQ����S���ڕ����֗���Î\����ra	����'C����8�W�t�����"���9�K���!�85>���.�F�SW���3L�r@�5_�X9:�x{��W��v����T�ϴF��9�Σ9�Z�߃�����"V��;�������|4O�(�����s�i���P�y��,;b�vMS��������(g�6��8vJ�:�ǁ�8*�^q���k��U�߻��8ΡL`:�Z.���y@��3E�ċ'��U��a�]�x�hZ��j)��
)[�Г��;�� ��+;Lv���۷Zt�����rna����D���@��E9��L�����WN���G`	�|�0 sY@���5��d���%�l\j�v�^O�l��8	kq�CI�c�𲼩2��;�q�9^����T+��)�G��&��mGbex�2B5��9l]�<�����x��
5`�V<��լ�ґ�{�O�O��d����u������������d{S9E�� [�sܷ���3i�����U�n�Ko1�������׊I�>�Y��bJ��kW�v���dY�+h�\з����F�8���mb|)��c���/��}�w�ʢ�sZ.�C�/�����Y���y��*a�=nR��;_}�w�H(y��1��WGM+X7�P�JzX�b�g�APg��T}Bu��P="%mG���*��	�<o29�%)��m�ezf���F2)v��g�ǀH|����������2a�0|\��-.�t
;����S��	©f'�ŗ��e���|��9���k�w�ٌ��&_�&���3ͩP���9��� �ݿ���%~�ѭ:t�Ы
6>ɉ�C�f#�	[�7R����C�ɇCr�j���}C���7�T���x���s�����o���� �      I   �  x��XMO�F=ï��C"�Z]���d��"��*�*�\������y�`<�,�b^���U��^��^ԫ&wUݞU��6ϛ�;�3�_�^�S�����*'|���E'"�4�ZYa4�IJ���8�!
�L��d�D��e���K^]V/��o9�N����$�ݲ�բ>]�&��G�W�U���$�BW
"F'c�7wgC�L鑱��Z4��zE�ja�Iey`DZ&쌈�� l���k�Y���,��I]�������m�-aS���(lHR��ϫ�<w]SΓ��|��]��r�W7�~��n��q8�Ϳ�6i�˶o����U���9�WNZߝ�~qZ����I�������:�b�D*z���Ew��z֣W�3o�bs޽6p��Vuܴ��Wu�����2�_u���鮖��g{��[=kM����|�y�f����>��zщ��\& ��E�lfxppץػu����A�����[���{���˪��Q�i��w�-\�[����	k�H�J�i��a�иk|�7�����f(O�Z����oͱ����G3Dnw��!i~��*�]M��p���������/sV?��)�����kC�i�k�܉a�s��lQ7�O����f�j1�Y>�1� w�����ޠ�%'�y���a�<��3���ɓ���xYr��Y�C�O�J�[I8��A�B������0,��(fGg���ض��F�ǈ���hp�A��T������rA���[��p�ņ&R�n,���:��0��J%�M��W�8�xi&0��؎��ޑV�<��u����BDA1��A�0����+}_�&���ēqR����XGh������2��!o����Y�?���	]҄[W�O�'�X��T�o;Q�R�]-З����� (|����-9Y�`w�p��k��q�ω���c�D�ݚ��[�?8�-I�.H@ƈV����!eJ�x� T+���jA4���O\�˹\�u�P�27q�7^8h�`�B8h�������Y�46'q���ZĔ����a�B5�Iuz,e��@�	��:�NM�pR��5j`VA�P輰�V��haM6*^��M�?0zD#�Ke��M��3}P�h�`��|,q?�PJI�_��-$�g��{�b���@!���`�!��X��L�L�v�����8��Y)�Hr7TNݦm�=-�v�p77�_�xV��b�w5�/^�Ą�mD�?WR?9�kG���gڷ�!!"5b��p�A��naY����
\o���*H�&�":m�tM2!L�w8 ')�qR(����g��j(��t����3؍G�5��"jş��b
	ҳ�@3j��6�^J�)U�p(N鲦QcY�M�!:���ІJ$%/"T�m�A��~��ŧ"(�Ψ������?]n�\      J   �  x���������;O����(��t�>&0ֈ�#;�$��~�|$��e�����X,ٟ~����߿}��ן�|�����������j�X�)��?|��d��.g9f�2�Z��6����>NmSg;��-�m�6�qX�ٴ{[�a���gه�{��ކ��b���^K����e�"����-�ܼO�>v�����>A4N�:Z��Ѐ~����{��Ѱ�8U�b]+��-�-�Ҭ��=�<��w;�s��
	��v�Y
��O_�����5�><~�wݾG3����َ]���*u�d齍��Y�kA�T�� �r�$�0�����&��rI�n���np7U�e�&h#h��/�����?�������J��$�"/�bW-Hc@�����}�ei��Q��p�0�iky~?4P����"EQl��$Y8���"Ŵ�C�T�3�%�C�BGID h�%� �U(K�y=�^�ŭ��k�0��3��O���bs�4ݳ�G�+�!���~S�q1�&����T�YL ��U��ӝ�+���EQ#ʽ- �R=��p���Ai����d�WN�+AԵBaQ�cUbqS�#C��1���,s}��7P43Ѳ o��fz�Ydr��H�6ޞ�`��R	�N����:ӞOF,���:K��B*�n����Y8eF��;m4L����*u���`i�T�/U��.�\+��6�<�<0ߥr
���O�+���|���%�ڧ���'A�����nX�Iyu0��Ô�{������o�ǫT��\��W=�1�F㢙K���b:���.��>8���'��t�fq߬�0rN�z|�嘀0�ޑ/Q��\cᕕMgD�7׭\ހ�g�bA�dh3������~ӯi�'O	g'�~����l��Q�N/9���q�W� ������d��zV�]�=J�3$��gq�l�J��ȹ�:U}�&s��ĎM۩�"��C�]�S��}�U=�H��.�+�
nT��$
��M�E�h���LMXfmZ���(����G�"���x�>�O�@��hr�J��q���s��C{�Ϩ�['���;)qj�B�s��hq�tE㠞+z�����J�#��ԕMa!�>�̂qJ(�t������HH['�O�9��$�8�� �ϓCK��L+kX��f�.[� �6LA]�qѼ� r���N<�xa�Q��t�o�.�~�sD��*Q��C41�\�1D�\�D�C˩�QF<}�̖�y`��s�/�J�>�eit48Ki�zy�A�ܟ޽�mP�Qp_��SOeѫ��r����$�	]n�킈��I0Z!�����{,o���)�*q�� {;|d�"W9�0`�̨%$���|��#н�2�7�Q�]~_Ԕ�����Ԑ�E{_#Z9�H�����Eͼ�og�0).�*� ~�k���;�"�������,�_���x���_����pA��;�`6oW�0�c��l	����'I[]z�4W�F>X2��n�?�w����k��TsN�|��ZH(� s�b���8\��o�W[�t֯�����k�p��D1"��nJ'�}�MRk��iq��.C� ����y�3�u���z���^�%�v��<��XCow���vUw	�閝��������QPGL�tP���O6���
��|o������?/�Ͻ      K   F  x�U�KjAD�է����Y���x3H�1zf|~E��-S�x$Q���O�m[�\O/�y;=�o���r�j<e���&�b`������5Lq�|�p�T��=28'���6�Q!��1c���G1��C�Rf�T-��<}�����s��.j���e	��a*��{�P�BW�I�N�v���s����t�2�R^(Z �ҫa wcNf�3[�xє������~�ܝ�\!�;ʁz��Ɓy`������v������z;���`,�z�D�BȆ�C�T�冀(R%��D�DT�mA
���I)���V{�Уf��O�Bg,?�,Q�|�      L   �  x�E���E���w�Qu�;DB�0�cY$��F���U�����LO]�K��z��×7j���~襱�v��ryԕ��b6γzl���|��v�F]�L��Ү��K3�Z/~����;�?֋�7�|�c�qt�c]�S���Zf�T���v����q�Wx��=;/b��bݲm_+V�`ݚ.6�~���d[��R�1M���Z�YFs*<�f�"4��!��lr�(U�Į��Z��2٭K'�O_�t|j>�$Òf�-��ԗ�Z4�EQ6k�U Egl�_�L�h<�� `�E�eY{�^���n��dT�94��P����ǩl���'t�m�ŦDR�q��(l�)��+Dk2�C�H�~�rCZRt�	c��^9hhɼpj��l����U����)�<�ѣ���-�ΩW����3~>ᰂ�Uy1���T�uGk�!
�i)H�ħ	J��`8h(�gt
��y (���������;�0	xI� cQ��+���+�!>��5��HG=F�@�-uiN�.����t�Q�.qW�0��@���HM:G�����$YO��ŷ���[��%���ȼۡ
zD����{��������]�8�����:]�:�3N>bLh�HT��.8_�6��;���ҍ:�誔3������	d�;#=E��l\��y3M��J��O����k�ё�( b�350Ԙfm�������O'+-B�D�֣���/RS���ށ	��=��x���Ť`6w�#�gE[��sfS��GKg�BL�6�z�BM��=B�=5�_A�ȅ�:���)2�6@z��u�"��=�x>C��l��$gkn��N��Vz���1��Xg>Τ����花uJ����'�0�������h�3ؙ~z���dDy=�Z��:� ��׳\aԜ ��Z��	��Et�v�AϘO�r�=���q8Ğ$�B���-���U��p^8�!���.��殘��F��a0���������:$��h�\�7�țo�����
��{�F��p8���h������9̘<�o�=��w��H�s�MI ��(��zW47�>�c����h��w&��.����͉�&������u��@`��n�=:_ur�O�ڝ~�.}���sc�x ����IHt����&ԟ��qpþk�[6�3}�xL�F���p�>�������w�<      M   �  x�=��n�1�g��Ȉ�Ɔ�RաRۡk�4��R�Q�D������o����{�{kt7x�s�d�pJI�{
�-�����l��Ymp�Vz��π+�M:��F�/�Ȋo���]��k�1���xHg��3��������[C�ɚ�ٌ�u���@,z������B�x���Lʀ?���J��,��L_��ɋ�wԛ���sg�{��Fo�ۿ�'�ϛ���L:K{��Cy&�������Y|l~MOx�;vߠ����t�������|`Gm�9�sf��"�]�G\��+���L�x���r������t�����Ag�����RܮO�q㹶��ڱ���z�}�������y�
�����{�
�8�u���#�??������      N   2   x���/-N�K)�4����|3sr2��9�9aL.'��	'������ �      P   �  x�}�KoE���_�G�����ci�A [#X�i�v�h<���@�=߭?"������sν��[��>>n�����qZV��+�26�VǶ���1�XW��*�_�nURU�y���)W�*��OV_E�ꛣ�ޭ.�y?}����p~����<MO�xR�f�<|���q4E�B�ZU��?q�J%k�pᬮ��XO�r�7�f��~'���r�_������`�zU�Y��l\�?M��a���z"+��	����vA�_�L�D�T�^��u�v2N��l��=�Ӈ��L�=��ޑ��1�H���'���ϵdUk�՜���>xO���~�U��6�L��v�/�[�Fe)a��Z��#j�VǼ�E%K�����j�Y*�"]چ�Չ���z������$IڤJ5ɇ��*�p��۠R�-KDI/)��|RD�[��-������,sk�2���o)��D�ƬR�:Y�'���͹ ΐ�$�L����{B���v?o?N������[�X �%K�\i�[ �4�h�F��hJ�J&�P����_=�n>��}�֩����-u	��2?.4"a��v���O�|K��{���)�e�{��S�V��yڼ�d����k �u
eb0*��JB������klka��B�:A��ð�y;�ݸ]n�؄=cK.ʗv�ȩ��g�@��������q�OÇyy�_��.`�$�ɥ�
�H�I:�2V�o[z��Ҏ��5�ߍ7�f��°��Q 葂X����K�`�]I�T2���9:]�ٟ�busTXW/B��QM�ƉT#��f��?zH#*��v�q�R�/Q.}}A��E�*}��uIa))�'����:%����HM4�G�E���Z�]��q?��qө�<6QZ�<&�}��ʥ��s��H��E�R/ &��0rC���I\���8�4��7���E� �U�O�m:Is"���Q�� ����T� ���
��K
jX�W-�W>]y���Al]l�J]Q}����e���Y�,�q{���6|H@I�BgY�!ǵS���-0��E�h�/M�P�g:C�iK{���G�v�S��d4_
� �!�_����� 4��9
�����E��xq%�B�1��+��w��審E���``�c��Ki��G6;f6X1Y�SY�7P�(*#�A�+�q�cT<Y'�Ay��D�za��=L?L���P^�-�ĉ��bԵ�ejz-K�{:�(�S�0�ϻ5�з��e�����Wµ�(l1�6�y���0&��/�O/�˫����?��~8�\_�������_�x~v�x(�ѣ
�"��/�LT��UFnb�ۨ�=Ŗۢ|E@o \y��f3�td����@�6|����O9�z�|�Mjd�y�:ʖF�����V�F�gۦ-Ϗ�������|��4��@Q�g>��߷��,����h�4��\����f�����t"lS���K�2�����cd���&��$#��V���2){�_4��F� [ġiU����q(c���Y�z���7��*Y�dD���7g��r+p����F�6Vr۶%+�dٵ2�z&����Ge̢Hp98ٰb���b!ۙ��&96y�V֞h�_E��Q=����������.�墋ku�6����<t��%3�t�I��,�d�Ʃ����^\�^��[��%iQF={��"Y���K��\�G1�L�e��:�.�$㉪ʜ������0O�i      O     x��W�n�F}���*`fw���j�KE��FY;,hJ����Y*��ة
��2�3gF�����7��>���h��R�b���&
k}1%%�N��"�(T�G2�x#T�b^��&�e�},N]�k��8����E y^��ş
5�j�!},i�P����kJ��Ql���M�Mm`��A�����J-� ���~"ႇvRU��~��~��m��V~]x����_ܰ��N�����ߗ4)-b`7�Wְ���Sڔ��+t�Hx"��sQ�E�[��E'�ւ챨��U��S'	Gl�d��s�u� ��ogPa4!�t�&
�کq�1��|�`���-$d,�`)�S�x�*Rl��;��P`QP � 2�m�p�9�ؖ��x(�j��/��dUS���p�lVw��>m>7K�3��Rm�p'��Q� v�hO`���X��q��T���|�6I�8�BS�6M.W��t���~��� Cj,x�W�&X�%��+KK!����N��?����J)6�fw�)$����."�# .Rގ���G��f��"��b�����ơ6����&��6)F�F>$�7(�VS�6ɦ���Z��d�w�4˸��S[�*:6�*ջ�Uӥr��]�Ƃq�@J��>p͟g"3�ɍE�=X��[x$Y+�9A@� P<!#�ϺR���׫� ǨW�q�I/��X����9\�A��V�"��婱�*d%��k�l�f�z`��� /d!�Z��9�M<��i6���1�B�ᏁMhx8�E�}�Y�2�h��t�9t�(9��C�D$GD|�t�sF��C�4u;�O������1qe���A�C��5��Ts�;�� �b��(�ۗE�s�)=4Klp 5"����r=��DI�� �h`P���944��x�G�e���bE���-��+����,6����1l9�6z
����}(@}  <A�b�*J|�D�ո�P����=�W�b&{{�m�o@153L����&�"�ր�-���ks��oը}�%��hk��߂�G�� ؼ�Hxρ'7V�F�E�k"l[˾@��T0�g��>qX92�&SE�N�6��*�xd�0)q�J�5�ڣ�E��O���#�\�2*m����8�@+��bt�Z��oK�2r{�v�zl茢Å�������⪼���^,޼�����/��׫�������b�@��0� ��l۴S�����B�c���Ž�>�����ubt�%j�o<ٗgM�l�����3�����O�Ѧ2�W��u[.ΝA2�!�徜�NWov7�Uγ	�M���
��ݣ�I�|�z��Ƴ�M��e%%k̙Ǡ0X�2��4L�|jx��̃;+|����@� �ZTٳ�gW�WH�<*'L~�Jm><�	׌RLzA�fn��$)G�0')�3�<O�HfV�<�3��ˋ�7�})��ڰ?L�����n#��a�Hc��z@-���$�c��8C��%I�D�0竾���l�8�`��9u���G�S���Z<2���� ����z4��ꪜ�u3{7��3�[��Me%'��d�/��b?      Q   �  x���Ko\��׭_1�x���ս�!^q@'�dC��	#	R��}��C�cC�" �w眪�����x�p:��p��������fv�"�\k&[37炾DԑV�mw��6,�WO�3�%�H1��g]K�(��؛뷯��|���k=�>�嗑K[_���r���wFn�p����\�������������p�閌���d�R��[��,Z
��m�+yr���H6�=��#)��쫧�PO���)����т�������S��<-�'/d�j���iDO}���ߟ���#wZ�+��{�#�`�m���X섥���Ө����G��:��y��#O������.#'��:��λ�
��b�(��G�Ro�[�Z�<٘��)��4�f�Sϗ��&�Uێ��{�^���y.uϦ,����gS�}�1�FÊǖ����Q�f�Ȁܯ�	Nm��F��@�r�Q#%��8�*-=+�Dj#/���Fd�R�h@�).��*o_�nNO����������~�B����N��ڝ{�-�T�9�������+(M����OyԖ�P��E�J�7W+a�2��K���?��*�(Q�ŌVU;tG��+_� �A�T�����P���QL��J+��E�v�.�\����R'�������
�������"I���t5��Rsx{m_��!�e��B��z����.���{�57����v�Y\
}���oU��<}{�Zi�,�z�Ո*��@F]0V�+1�j�њ�4
Z�,��2g�YHT��H_�xw���q�,d�:�U��/��_�(T��T�;�;
2�ֶ�7?�sK�0��� @�PR>�k��hXb��|33�`n��,9���Ѥch���a��Ң&
 ���I���HS}�c�"�ԗ�D�J,��
�Ae��"��b�t��M�����7ѣ�H%�8��NWC�e�� ����p��������x����_�
[��4��CR,�������ȶ��J�[V�L��w[�4�"�
���62r	W ����3�5��Yzy���I+�M9�W2�^Ϊ'��L��=gp��y:�iva��x;\���D�B��b!r�(.(1�	�T�S��=�^n�Q$�|�G/�h����5O�"7�R&�9[�
 X�+�i:L���}�ʍ�j��'�2���jR%<O�\�8�K��N��/-|������_�D�g���7̡��iH�`PMlU�ˌM�ΰq�o��3�	�6(歟w+�}�z}IU�!I���w8��iWV��K�֗��v�PR��[.ei�m�h�Ƭp�%s,�X%O�Y���
����V���*/ځ_�������6DW	���Tኦc4�m���|V�+���<*ԓ�+���:��6J���w�Q��T�i�j��U0#�I]��������ꊆ���')�l��X���$7&Z�ʱ�+5R�M.%������Q���.���j��x�>>~��%���{���X��P�6��7���#o��h�AZ�n[;�>�Z[�a�������ʳ_8����-���B1��}��t�}�bl�� ��E��ei`d{s�6-����=�A���A�-ݶ�P��| KOi�U#��Ҽ�B�����>6�G&n��V��(�����������xS�����������^}:=�h�?v�� #�Х[{����1YI�68�o�i��K�Vm��-*&�L��
`/���*��ٿ�nw��7h�7���i��Z�M;��%ۡ����Žm�*̋���ǽ�b�7���Q�r,-Tj[Ӻ$`kl��"��L�I��E{�۞�p�l�]h7Ԙ����O��uFe�� pjJHe�������ѿ�A��̴�2N�~������㏇��}�tJW�H��ѧ_���t��|r�������EeǷ����]$R~�O��3�z���f      R   �   x�UϽ
�`����*���믛 �:�����"�+x�j�d}9�,K��iżaơ!�3V��|�'��~F2.=o��q�?���0������4�m����w�/yB�ga���a�yz�!�)�s���uMD_��6�      S   *  x���Kn�@��0�������#��YN$K��P4�K,B�%}�����Z���Z�?#��;��O�8O�mu�iו���
qc�F쀸4&�q)�+y0�y��86z1u���.'�[|ȹ'p��W��L��2��z6Y����h@͇mŸ	�ժ"a��5
a��dͲ����U������܌2cǔ���"���\=�m߾�0����=9���|~�~�c�qh'7�kCI�z'%�Q��8kzsN���K��+.m����ϖ��@^�h�®�A��*j�n����d�$]�R:x�|�y�_c7�$      U   
  x��Z�n�H}N�B��.@}�<f2�@������C'9��ٝ��s�e��(��a5MUߪN���_W�ͫ����~�l�����ͺ�Ym7�ޭ������Us�mq�����_M����K�y����{�vD�e��ܭ�����w������lV�)|�q�u��.��?W�����z�Zi���f����|�������_�77����b���St�k���ŻպYl���Ms����l�������_��P]^�nWۿ^Fۧ-��F��_n����vS-.|�~�o+���k�mu�?�z�)��_���V<Wk�N��+ܬ9���T�a���_���{��4�.`V�toa�#����6��/ņ�~����m�1�ǥ�Zo~�6`'��/X�������1�_m��������>�o��N����O|��^u�L�w�)��}������bqyv���g�_����2$��M1��7c&>����4����d~?G���N&��|�� 4��TD�zD�<���bS��A�)z���)�&���Q� ��O��_��>~|���0��J���==/��6�T?Mg�>
G�z8ɡ���t�g��p�g4Nw�g=�p�F2�j�������ӛw�W�/����8{s���W�m�a�3�p���9U��]�����-�~9���8������΅�M/��}'m툦�{:����Q5�I����O��"�l����Y_��9��8sz�N��ԟH{��������q=s���#�㸢Y z���j�:b�9n:qXW1�:�^c_:��t�fjU)�[��nckɋ���j���q�^��0���=4ڒS�%P�����甯B�u��q�UҺ���r�5lV��F^v�*�s��hwe��/����V:V)�:ŧW�?��ƫ:��ވ�}~@a�Le����z]im�̕�J������.��te�r��znN���+�#~�mǼT��d-c��ɻ��R�b��=�_=ٟ�|��~�6׺��|k�=|іT
u��랏�����W?%�a���Z�S��M�l�V9��J��.W+ĭkK�i�ur?�=��'"^�%��w�U{�e��B���K'1�*�P�eT��v8e[��;F���ʕQL嘱�RV��^���ڣ7��UtK�����6�:����k����ө�� A�/��*GU�ؓ��8������}�g����h�
¥��uP}9� Bm�q#h�E0y�����Q0�-f�L�>k�Xd��^?!ڨ��0F���:c%����K�7	�V^��LӉ���c/S"�z�'�R�����0
�X��kE��d��b���G:�ڨ� Ħچ�������A�� ��彗OL�XH�{�!Qܴ��~U���_�V�\���@�����R�
#�3q�V���f%t�Z ���!���DK�Q"��ek:���E[�B4@Փ�	J驽�`Ic�2�@�H�r�= "�J�w0����v�Ƣ�(����Uk[ͩVDCp2�a�]����r>�0H�T���� %l;�^H Ps���RQ%	��"6�$�ҕdJ�!�.��ÑC��Toa�N»���c��9&�J;`�'�Ռ'�LU�D�	���i�,��}�c(f���X����P2��}Jg����[[����,i�PZgk������Dg��A'��=\��̲#�v��p�S(2��J�u �l�3���t�a�)x,��u��IH�7v���K�#T�����%�r�/\�Ob�{z�B6g�0w���Q��\r��j-QU���X�s��[g����ղ�9T	����h��Е�I��ג'E+ �R���NȾ'���t�IΒ+�v��~��Ƙ��(�Ȳ-�+�G��!����)1�����/۲W�7 .=�~A�cu�:�����FK���t��i���l�'��!r^|����!�¶�S@cf1�Kc3l:��~j�TV#h�`�}�2��%re����"<����S$�#+S�Q�J����I(�s�r!�L.`�6!&9)W�V�)|92�*e>��?C���IF.�n$d9��k�Y!�v�6c;��nF(�'���e��!SFF�P��-s����1˙K�,��)'֤A$b����]L׆(�G�����I�1�z.�5��d:d��E6�W�T���� ���O�؉�Q��,�M�Njp�L'� فX�$D�����i�4�0�È�C:�eX�-^�,�/�ծ��T��ġ�6�H�	�D! ��Z�: �H������	��ө���o��2��A�Y ��XyШR���R췊E!čE�8�ݕ�Ip8�lK툎'y�.i�/8�'b�,��Z�q���x8'�n��&�rR`���� <�ۇd��7�.�y�Ȧ�@�v�*`�0�7ȡ��r�'�v�+o#ˤ'��p���Ѿ�Q+I���}��� 2M�d��I-g��k�$��hb�7��'QJ���t<7M����I�NxL���'��I�T9,��R�ba 7�t��V�)�Ulo���b��N�(�5���s= Y��� �Rn�pA�靷X&�����ϸ�$�'��R�����ֽ��TGlF�jV�i4}b�I�M�:3)
��n�p+�\��������D;�r�";�%<+$e��THa�L��f�n�~�z֐!���q�<��L�<��s�<�j�cA��܁�KdĚ'�s�`�jGK	���˖l@D�����E`�)�`XLPX�3X�L��n S:֖���j5�/�#)�r-;"�"z�3����6��xm�P� ������1Bi�b�(e�I�� ��q��@OQ:�Ƿ~�a$���7��壒g%��������ܽ�>�Cb�F?���q��� !��\<�����
&�:�6��$�`-F�o+\V��a�[J޾0�
��j��X�N��	<�i�*��^< �Œ*!(x�'R3D|%����[�G���
S�d�N����������|�zY�s1f103���a��wI���eʾ�x�R�>3{���aj<�ֻ#D�Xi��%��jc�d?�:`�1��p������T�-O	a��W�UΖ��^1Ve����Q��Jf�Ք��J&V�A���sFI��*,�##Y!���W!{^甕T	�$`<JC�+��@iV�x�%3�"83��'6޶��KH�)Wxf΃�,�S/Z`#	)�Y�]��oP�9�t�A��-V2����P�W�1��+�-�)%aa8w�"�ݩL�ˈ�f�e}&򳆯$���a�ȯ˝xk
jwݠ-��$=o\q+��s���#o!,�ݰ o�ډӶe����6o,���x[����8J9�hVp��Y�fʌ�M�ӕ�4�z��Iw ����ƒN<��04l8K�\v�N@2��P��%%�yM��.�¥y�S��KL�%B_�tE�S�?:�v[�h��y�O�6�z�t~��~�b��+M��T9���#+�L�Z�<N<�#^'�;�p�¬gg�.9{S��9WN�wO��V;��?��ׯ���D|      V   2   x�s
U�K�M��q�Q�W��u
U.��M,�T�v���rp��qqq 7�|      X      x��\[�Ƕ~v~�P�H�>u�<���@f��-�3Ӏ�1�=A�_�o��v`��)a���]����[�����O_<�^�z��ٿ'֘l���L���n};=[u���l�u�g���j���v��|�/.~�80Npf��f��N�|�YO�������vn��M3���4�Y4e.�,'�7�:�y��2&ͫ�U��<�M�%�1��8Ǔ�f�
�>~�6����������o�M�->t�D.���M��m��w��ݪ_��<a���k#���wn�At`o���j������f�>vd2	$O���nW��_��[P�\��0�0��:�Y�2�)����l�9���������
_�r���:r�g0�ō	2i1�]�/,�td�l���^�ޮ7������by��3T7��e�g�`�c��K���Uc90��#�f�|���H�9����-��a���ܲ)�B��%�}i�?b ���)+d'B��(/d��כ������p�������!0ѓ�z�o�o��5������)"�eV(������Cg���������>{z�������/�?��r��R���Dn���M�{��aM����1޿w~�Ja����=����6�GO/.ϟ������u6Q��oX��M�ݍ�{��7uf�,�:�ntw`1�]_?:<.t6��v��L��[�����w�2��塁�ӱQ���0=�J�a/�C�vӇ�76;������T2�.1Tu�s�4n٫��_�v�?�G�N+B��/>56���D��x���wW��V���!|���qyH�i��Á������o�������&�����4MUL���CcO�'g��z��4
��f��%�</y�9ğC�#��o�woWF#&���>.�o�7t<>�K�����n����㼍Yʼ6�v���X�w�}�z|TJ���n�a1}t1��6��
�/fwȸ�M/o�v��]?d��l��f/6��Zԡ�����y%E)��4+����-����54����zۏis�&̐j��4��]C���I�0̤���͇a�d�=�����ݡA�]���$&���NF��{�]AJ^�WN^߈���`�������'�w�����uV��!�
�Z����O~�V��e�iO)�@	�Zڢ{ �,P}�=l&��'�_6����%df8��t]C������w��C���/��~h��xV�T���=)~w���]OMU�ݭ:J�~X ��+�{����o�m��Đ�/��/�8�"�LQV�I��{��РC����m6�e��{�,�g׃�������CC�:��N�V�͇-��~\x������aYޤC�j��G��'Sb<�3]q;�;0��1���j���7�Ş��4��{�ϛC4O��&�n��v������И�K����̻C�i������;�u�)@�t�� �4#�nQ��|b�0rTN�3W}U2<��)������Ĩq���z�>��N��h��btn���KC��	c>�������3F��m8���ywpԡ�A3�YlF��3m=�O��c�b����������'S���`D�q]~cD���xu�|��㰠X(��,1mt��Ck��||Xد_�6���Kh��y?���l8\�i���H�W�;}��=����N���5�G��5�No��?�N'F�_���J�� ?a#*6��v����o��Ĉ��]��z6ڮJ3�on�����h�ˏ��/.�O�׷���b��y���kRg���Fy�;4���� 5'�� �^մ������rwH컞�C��ח�a���<LR��k�<��z{h�}g��8��������?��=�БSd��y�[��Oo��%��H����Wg#y�;Xɿ��~��������zy�n�d�~�by���ӑ1
���>}�'rS�C^Ïc_fߣ�l=�ϑ����Yl���m�-��+U���u�����n�آ�V�?��z�3����z(D���!��e��FM�Lqnf����)�/86Ag&�f\eh�Ż����qVe�W�@e��Kݏ�n�#if��<���g�p�^�����;'��7 �ϻ����*�{:�0�U�7�[�����C�����]K�=Yo�����	�b����d�^�-����^���K}�Ȅ��?�m�l}%�D�c��X=۲g�<�x�2�>�6��h�1'sp& �I���ۮ�6WfN�6��G0���-���g&sV|�]��7�jh�G;����<��g���E��ky�M�q�λ��_��^�,��n ��P�f�Qkœ�%�[C0�/2�ZHb�3R.VStFsvV�qp�}��H	����?����'*K�uo�/�O_�}�yo${� 4�O��+~��Y�Nc�er��ȝ�(D��%��e��(����@nB��g$�� �K�(��������o��ʋL���W�/��6�0}���G���F�`���J3�}�"@#+�U����h*S5b*m6P�G�~Y�uh�)�<�U.�Qw�����{�D��&͸2b�LO��(�r�=��(ǩ
ܩ��z4�w]e3���k͙�p3� Y��*�u@���_������Y
Np���L�I�է0+�J.�㝥h��3�}�M�s�$��К�9�d�!�	2��=M�S$��x��H��U 7PP�Dxa���A���~K:
�Χ5�\�"���$2�
�:y�>!-(�� �x���A��0'p?I�hxz���]�L|`:�Jn �Y=6����%TTBg�	:x,�a^�5�y̛N@`�)i 4U�����za_� ���e�$��kOO>�?�
,4TKEUTX��r������C�+�Ho�FL$!�a�7�E�$����� 5�H&A�|��8-F�V-�������-��*FH�,��kf��s�*���:A�����T�D�0\Vrb�Dv2@+��]2�ˌ7U��P 摐tqWOSC��{�i*[����c �����@(�P5�����d�c6�v��Z��$ǷFy^X��6ܿ��Q2uz/��2p�۫�Z�p� ��wr!vy�~;����q�Y�r����uh�
q���He-43��.�Ab�He�I����P�V���E�kO��J��a��N��j�+�f��=Z�⨆�?݂�nqCm�^n��g9,"y�p��d�@�&���m2e%˵F1G�%��:!�V��P~$?b�`RE���C���	}�}+�,0*��x��w�]���_�w�k���b	�(`l�&g�j[�}����b��C���a�j�ZDՁ���O$����ځ��D���ȃ\�r����4ۗ�	T/��&D�AP�U�@��EDT�yL���I�Dф1$�@Xn���zI�i��K�B(�H8Ú�fg�>ݏ@�z�~�v�����֫����lK5de�L[He?j�9T)�Bb܍�З����k�Q�H�k	��M�O�UiZ�+X
*u訙o&�����RE.{����ˇA��.i� X8j��:\V�,^ &�F�W�Z�6�%�Lܐ�:����δ,<s_`�,�k�!J+��0傧
5���^!��GiPӾw�:DCK�T��s��5�yݬeH��,���4��_�DK�x��d<d��j��Ǖ�?��-P#��
��������!}��`�e&,�T��FV�E]+,L��VQQQ]	mԇ,��4�zrR�'UU����Q�Xjh�M�|����g�����^�:��+�c�t���N�f�X���
�%��o��0=uJdD>��x/z0s�c��(kQ��K���h^��Lu��fqK$���x�.�R�@��k}OoN�\�A��Q�-(��-
��%EU	:��O9H+��Ab��V"1T��hY����U�(�eĺf��&FY��� A����S��7�(�\��$)Hw ���jk����07^:T�3�Z��F*�*�a�Eh_�m8JI"���i��2����W��2E	�[)+�꘱��3    �M����A�]Y��8�^B���gbA�0��Rh�_���k-�t:J ���a�GKA��%�g��(W�z,�+V�Q�w;���?�=%�*<Z\,ō��k$C��X��)�t� dMa#V��� �\N��(��d��T]=�FQ[+%A=hKa�d8��U��\��M�.1&������f}�[.�7�nF(�{9	M�t�ߢ5�"B�J��;�N��틴�Q=�� s��i̐V(+�:���,�P����TU1�n\�m�j��f��W�ܫ�uQMET�pGc#����jW깫�>%^�$����R���<p��=,�&h���693H��9���4Im�M��h6'������(��
�d�P}3j�L'�Б8:[.��z��p���uɊ�\����C��y�2����j�\�l�(%EA*����%�N�0��@\a�npZm�6�.Z+���Sd�ګ�C���13QD�tʭp��Ol�8D1Xz��%!�͈���S+�� �r�C�0tZ��+�B�������I<cR�\TE1̤�쨍AL�P�hp�h]���ՄѾɰ����y9x���0U$|&��5B]�H�·ږ*�*�����,�y��$���Uc<0���d0�Ri&���8��L&t�$����,
�gi�ثW r�o#P�M����%�gh�Va`��|��
b-�_���a�Su���kp���J���q�?������Z,�@��I1�ř��ub��NNi3�9�b'���xwV��.�M�H�S�DSEC���d�����Ä1��a�o�:r��@�s������BS��k	n�7
��r$
n���� %l��I���TP �ax�g��'V�[�OFB%��s%P`���wQ龢r|�(r�@*�C�J�J�"RD�#^0���F;��9��Ē��@p`w�2b����v[L�+N�&u���	��k)[h��s�{��R��[أ�hܙ����m��w�EO�p�3D,��G���t�y2���v)Ø҂vUN�yaz9b%��n�@�p׍� G.L�p�H�
2�>2c�>������)>)��^ ��b�ȑ� ��M��K�
%�l&E���a8���P�3!���W��5%��#�N(
�D�y�@	�*<��"S�&迾"��F��������BV�`�W����]��K7�0$'��l#�닣�(��4t.ާ��E")��� :$$��8ǃP�b Y�"A��{�����i��*J8(1�-��0.*�
G�3wV���0��Z�(�e�����>���S�K�S�Q�ADh�m�t�����"> � �B��ӵh�ZIA��]�ń5�� (�p�]Q���=��"3Ci�frr���<�/��I�0�h.��"�Ԯ���7��+ƷJ y�{���*�8:9�7�c[(%�A�D�f�(;�!�oa���$OeN�،~ �0�f��Ƭ�];�-=J�)em�g#�^��հ�:%����������?��n���}�M��77����d�GнӘP�u�ȓ/��̌G�(�ʕ{���c7�Ь޹�B�(%�J�^�rM'͵v"A��״��?�k8f����(����%��:7�T�Te�| �.'h3�����@~�r��M���|-9P�>1_n�1��G��i��a.�B�ێTTx��~6���
����D�{�][��I��@�i)�[���N��%�~�*�⳨�`ޔ�kI��/�\Z��U��
^��ٱT��m�a����>�f9�}��5�.7�e���꽤�`%`�z¤�	t�3���X�"j_�r�(�X&T尮�2�
�E�h�_F=���i��KR�"��WA��H�_��c�l���C8�+M���v�U��f�m�1�|0G`�R \8�7������ Λ�̘�<[hk�V���5�Lf9�l;��m�d�:rS�Y%��o�ͻ���s_ư�*�D�IB'N���p�0��D}b*b��Jd��*ґ��yw	 �,.�^����ׄ�ŵx!8V�Dv��ʊ��s���S���H�-+�[@^<V���bfM�Jd�ġYsp���	�v���FH�mp�� IS	�[�.�i�G�~��NI����[�����4� (�P�ئ(ԕg���L06���(T�~��1*xa*�Mj�&�nK{C�_^3��sB�d	�*n�?A��<wց~Zo>�	{���-'�9�jAȠ4�l/'C��C]��y��⌒3ȚPd!$W �|߂�I@L�"֧��2|ɓ��7 ��m�f���B2��x�?��;EH��*�ˀ�1�*�M�8Z�b�B`��'yG��b���Nr��%ក�&BB���`�^�mIVl�/C�C�Ub5v{��Z����Ő��_N�	�(9q����J�tP?<3��=�D�~�~P����d5�+���*ڦ�'�B���HA�aP��,3Kr�=2#�㔁R[ܲ��O�&�,�_(A)��zV��!. 'z������Hv��%觏"��"Z�a��.��|8��Ȍ���o��4-����"�Oi��^ ԁ ��xZ��H�>fծ�/=�;=7�W��IÞ8L�%5�I|��Li�@��@RPJx���%D�H�N�7�^y���=\o~��:v�~D��|sG�7}��vRAĚ��?iF���FtX�
�@b��i6J"�\�ҀTeN*UCf������,V�k�J�,���ށ�,�р	DL�||=r�@�����8���H�L�iRx��:�:��0�i�K,���`J.G�99hVdsŢڢ%b]Y\%�u���3$c�m��Ir�p��bӾV_����bL EY���0?����sa�����������B��_Lb�(^��N�BCZ^=�p��5)IVVe6N*�0n��v:H�7��	�7;p']�*{�q�!@g�.	̬�sa�uo����#U�L`� �T�Z����s�YN�41���+U����~_��� <~�t	��µ2,*������+jw��I�
@[᷉	�F�䪂�,�V%k��VC8��D��w�v�;��b�#���	�_{d�N��tw��O~����8�Z:9��S)&�Zl�3��Q=<���7Y-1S�e��}#kFn������'�Q3)�t���cR������v�?N�q���J2(i���H��7K�\�
 j*��hs�e!D�2Eq�5���AiO��I*A�IH�J�^���9�~�0]AKW�v$c�!3ç�ﬆ��Dc��V�m%0|���Z��Y��VE��z�7M��+H[[n�ym�P��[�"�﮵��*�t�HS��Z~�gs'�ب�X�3^"���K�"$����nTe�>b����V#oV�|=9��B
��Y=�|b�B�.�E��Z��r���PR�2�^����5M��d��/,Ϳe�i�b�x66��r��*�Mb	U�&����M��j�W��j�Yz��W�X�GkF�
�S"CI>&�5uX��Q%4�#M���3�jiG�Zp��)z�'d�!�U���TRG�jp���j��R�g�jԏCp���ʷ�L��=�UC{:��<d:~|
��vR"�5+Esn�@n¹I%BZ4�G��o6�7Y'���'q#.����n}Kw��\΀h��,]à`����H�Ҵړ_5����y+�82������3��2�,�H��3���RCk�Ȧ�������J�82�<@��AF3
E�@�J�<V�9V�i B��iO~��^ �b��NbB/v��]򓆭_\��0����,����:�'���6z-.����<�7���o��]{m�VN_�z���<o�ٳ�������>�������^&�K����� �R�'= 8���V���B`�) ^��K�A��ն��L�J��N�����������5�q�}\-�:�H������f���H�sa�R�J�'�Ѐ@�Z$�������^L�0�.��'Y�)I���S�l�z�&�)ƪG-��C��X�[ܚų}�&g�X�ۣ$�ɓ妛�8��t)�`�K��^�G��SYIK�9��2P	�� �  ��uh���3�-���Z����i��	A�
��~�4��b�]� u����cd�I�d���O�x��^i�E��)�	�����ufFj�+�?��(�B,;,��.0��i�B�]��i�b.Q�M,��c������ٽ_�P��� 1A �# �x�%&m#c�u�L�F��&̜֮��f�-��~�ES�,�Ȋ����ި%�Ѫ�V��J�ws��� JO�c���zy�I#�������O�5����]����^��2<J<����λ�*�ڒ$�'2P�Jv�#���|V˰�"^'����r�$����?	�k�Sx���E�U��XNF'��,"��U�-�j/0;U5���g�Yǈ����F~�^�~S'3�b�R3�\������Ttk�4j���z�B���r,E�������Nz*��@��+j�H��{��?��n�      W   �  x��WMo�6=s�.̆�G�u�nld]��^�5�����Jv��7����$�)��҈;��y�8��}��o�m���I=;0�K�ف��[���Ra4�٫�U0��]j���~`0�|��2�.6C�V����WuwU���t��B����Uu������]�yLiZ��BZ ,Ê�|T�4`�Z
�%�Њ�><�V��7&���f�0��>$���ҒoX�f�g�Άi[�Տpʣ�_�-m�F���4ŤX*r?��t �������m�]e��cn�:�es�ey�!�OJ�Cʛf�]��AX�绸/6Ǧ�̆@9������"�Jpi(d�����L�E�O:�'����:�}�3ԧ�C�&8$mA C�&#z�U�^6Y���^8$�����C�cW��EdǛnh���U?���x�-��g-$���ء�2^S5D��3� �`O֧87�@��-D{�"�0]�>��Q�<`I����~=`b�vݤ��*��"b�7��X����6c]�,fg�rJ�g�$ �;Cu�t��u�_�Bp�2O�Ϭ��U!����܃�q�w6҃W��I3k9��Q�C����nӿ`!*˃>d�)!X)6������ǬTs7ð��u�6��\I:�I���ޥTkL-CA��k�e�����3��
E�.⥏�I] G���җ����Vi��m���>�k��Y�� w�"��:���L䛌�;C��B��ع��j��3��t۬�aG�8A��<�L�K�r��"�$+����~����q7Eu��RK�W9�T��DS1_����bf�^~�il:���zӡfRG;��Mفx�O�ݛ^g�T��k�v���Ԥ#�9� F�W�-7���$��<���
CZ��ď�ȗ�z�G�H�L�,m�,�H�#����|��]���P6� �W_�ڦK$�?ܳW_��Ƴ���{(T���S�m�����U���e�a�W�r  X�������{��o霻N�W�U�b��A
�@�P�CTUF��S������&�=�m݂gG����������~���V���Mu���ד�1��@�d1z�S�"!�x�$�bT��&��.��Q�tp3 �H߶i]h�ӏ�Bi7D�	 '-.L�����]�b�nH"��þz�t�vY�ϛ����'����~5K��������l����ʷ�orʳN���'̏�h����7x�Kuf8�����C|H��"Y-I�����9D�CttSI��j�֖�*�M=q���׭�U�'{p:>��rO:�'Ύ�������F<�!�kr���Vٜ=#�3W��Ph��<3%�ɨъ�X�>n��U�sp�DP�;���A^�>�?$5I�-��Ɉ��t��O��΁�O���V'���[����"��W9P�(=Z�|�r(�|o���g�[�qŬL�]R�<w�x�2���b���&[      Y      x��}YoI��3�W�A�8��/�nٽ�c�{,��Z�m�iѠ�6|~��/"�de���R6��_)Y,E�{�߿8{�n���ӳw����ٯo�9�JE�Uj��|�8_ͮ�ǿ|�X-G����'�'�s6:R����>�?�?��G�3D�����ٗ�n�����V�x��������F����L���_��H�J��V����b�������__5$Ɍ�M��}ȱ�w���狿��z��=�Y4%�ԃԜ"?z9��^�?�\ϗ��WW���ы�ϋ�������)����b�Z~kNI����az�GkJL�L�ί��1���ZRc���������˖d�ѫ������-v�u���F�I���\^\�%ď\��<ƀ�g���)1N�| b^�9c��u3�0;���[C��(�BS[B����b1�s>�֘7�q�\~�8��ܘ_�����_]/f�؍I!��yz�w��D��V��<EW��ɣ��t���qM	!�m-��G����~��kK�����.�o�7��d�6&ǎ~�b]���n�Җ?�����jNjv[B�����j~�\��(�2]5�"�^//i(h5�$��LWW׭lԣoV�W�߈ݏ�5%�t�����)�*m)�}��� �ے�F?OWl��f,��]�.��>iL
��׫�S���Y^^�/�����+��t�jJX�����gMi��x7�tC��������h��hDʡ�E+�/ڑ�3xю����f4�^�#h_�E���詃�(���(���(��hFM?xю�~���E;2��E;Bv/�s xю�*xю�:xю�A�);���/ڑR/�Q�#xю�*xь�:xю�:xю����v����v����v�T��v����v�l/�QQ/�Q�^4�c�hGʾ�E;��/ڑ�#xю�A�)����-xю�a�-Բ����p N�>�>{�ܔ��4���^}��>�.Ͽ�Og�?��D@RvDsZ�����t��l��k��8KjA����sL�x�<�^Ϛ���㚤����r�Gh��0����+�Y)��Wb�O�ZN~?={���:}��߿������/A�e��F'7���N��-��]=C�9H�S��������z��b'�wg��������p���wD�۟Ƨ�������Z��˛��l��8>���u�/��0��0	���O3���L/�fFm�"�<�y���z�R��Txp��_i}c�^~�+��>�av6Ȭѻ����j|vC����������ύ�������jA���KxQ#b0A$4H�Oǧ_W��OM�q f=�������r�$`L�Ӗ;'@0�Fr�z���X���6�DM�t������'����-Q�\|\�.֎�F���404;-G������~r�[Cr�s��~Z.�?s��%�E�5�}��I[�������g�!���ݫgo^���X�<C�~s"ӗٳ��[	dc�󅖥�ؚ4��7ˏ��7�X-e�	���8���fy=��q
iO2��֟�1I����0����e����\�-�}�D<j��@�!Y7?M?����{JZ�^Z�b+����iGc/ܩ��x3_\<���O�"�32��6ԑ0����pD��g��NI��QS�l5��5�l��p�5&$
'!�֘���LmDD&"�^���zro�R4t������j�zr�֣��+�8�G�����N~���[QEV"�X1?OW�OO�W�}���^��<�OI��=�m���3ڄ����&�Ԟ�6D��!�g�A=�h#���6D���6�e�g�5�<�mHxFې��3ڂ�ݞ��=�OKţ�OI����)�;s�ck
;S�����d��������������뫟�ϖ��������7�?��g�'�_ҏ�_z*�n�]�k�ү�W_g��[Q��Լ�MهҊ؛������U+bz+���G$�=Q{�D�����%Q0�~��s5[��|�|���
#D�˛�N�X;q����K�OrLũ���WN����ֳ$���ŗu�$M�t��pP�ͩ����8�X~�6M��g>�g��S�_\�$h�l9A��b�'
f狧K��O�e�n6�����<8W��m���;���ꟾ=w������O��;}�<~=���xmA�n����2p��Y� �OL/�������O�n��������M���CJv����庸C�>����A��&ew�㹩;���� �����Z���Pʌ�S以�w��P��!Y�T��G�r��TK�G�R�/*��$��-7�l��iH$rg�هLc�����FG�~B����8g���0�x����1`W�'L�a�����&�l�\��-�� ��)mx���wA��q���D�0J֛�Fo������ tP����FMO��}$�uC�� 䛀߲� ���Ui'd�@_��9�In�q�R��n=�~8��46>��,����$*s+�÷ipW�z?�1AG��Sq�7���D����ȸ�i��>Y����������~�5�u�&���mq3�'�~^~���p���9ҙi���`:�!Ӆ�=tC��Ə��8�@#�3���4��Ƞ��a������>M�s�15K�2�&��('�4k3B0�㚿�-a�^�4�v@~@ķަ.ܯ�~��Չ؆��4� �b�&�Gڔ��#��{޵��!:^́Ѓ���E�o���Q���yә��C�+�	�K��`*�LQ�X'�齃�i��|Fl�-L��R �f��iccہ�a�'�L���ñV�},�uC����H��8ɼf��tVO8����������G�b������'�c��p`�DZ2>�c5�c̡�$~�P�׷�:�G �N�!�vL_��(�	��#�ۉ����
z��s�w�D�>̺��M�����A����y@�����<a�Yj�If�"O�Hd˿?(�:x}� ���TL 2�r�i��N�����<X���=�0��35	�j�� ���$t$zR����Q��T���8���1Z#�t��Q�˃�y��7�{@��Rw������e|���Y+�sH�� >g��4)'��|+J�x���L��%�t��b:ـ��F�Jǡ{'�˽��c��N���)�\o����t?���φk��sq[Y�菢��ҩ~h��4G�9y٥I"U���Z)�@�O�����~s�N�A��,�5r��l<�OJ�F?�]O	+��h���	B���+~9����V��@�R�LY��N=/�{������l���u$^��xǭo�[�n�H�@������ؘ1�ǂ^7�ģ�t�&8�1Y�A�\���ձc5����sc���Μ$-���I���vZf�E0���Zi��%"i:�MM�@��[4te�k�5�dGFlj�oz��ء�ۃ���Q�a)P_�6I���mY�d�+!�����O2r���.�|�����Vo�0U�i�,��䁴����Jf]�S��O�x,`�Rv�W��3N���B{�w�;�v��4��zp��z����\1���D���K�=���e�m6�(5���!t���*t�z��~SR����c���@��0��X�L�K�� ��ӏ�!K �_(ce;�ěT�;+������P=1��]"�sǑyLJT�>�fOk��s�Xa���}�t��g�?&��ю^\�e�ݦ�^�,�L��2��lcHbz�}^ޕ��)hG:A=A����y4���1As�
�r4[����h�=<�X�}/��_��oZ�C����e���1#q��I�,z;��>K���MZT5HQ���8�h��E>�5E
�^�wIwA��8ź�L��eZ#�Y!�6�`Ң����g'��t�gjkS�����B���Yd^�]I�F/�	���ĨH��^OW�7_'dU}�2����'A��_�!#�0F��pW����o�V�7j�>�\���Lv�E��s�nDzbk-��!.m��� m�M��,��/*Q�كFf$���@�L�>ƽ�a��q��$&Kb⬡peh��d�Y��Qd��1p���#_��H3�    
W���
ء��Yז=D�y���������6�;i����i���#f��G yv����@�	��`֖��5��{.��ܩm�,��:���θ��FM�����V����Z�]+VbV���H[���	G:ë��I��￀�Jkݖv��ພ������׊+��汐�"^4�&pϑ�c��@�����	o���Yt��`z��z���I��R	q���\bb���0�W�9y�#��m9���)[��Jĳ��y�D�%�J�}N�?胶��~��2�H�+\&x+6�b���7���tU��Xk4~��CC݊;�ɮL�[��&�E��1⚦I��G;�qL�
�7��ӏo���o6Ø��$��| ˊ{�/r��Ŏ��!�u�x$�u�A*��+N�G�!&�S�a�@��i��/akʈ�̿{����1��A2��rx��)�X0������,�<���(��#�!������Qل��d�֦ĵ}o��d�t'��'���J�px`^���c.�O����"�o8x"�@��;z��DO�ɲ/8�K�m�D�r�1��͚�u��Dl��!}���*.���	� RW���$fP|(� ����:��<a6�7�A[q�J0^9�b�U�����p,@Ћz%���\���?��ia��O:����q�[ܓȲo�C��ۼ>��qY�H'Nzz�;�j�)�ګ�G�s� �,4I^Ú�hؤ׷�m����ÿ�zT�-]�O�mq��l(��~�������z,�us-�]�J�$�c�w�3���i�H�c����ɟ�����X������|���0t��A4��AX7����s���
�EI"[~J��	[�Je��-�ƕ�������IT��o=���$s��6�{����nӲ[*��L����B}�] $y�C杸�l��'vυ"�eh+܃,H��%�!����Z�p���='"�5�=��E��u� YIs�����+F��9
��ݓ��w��Ȓ�M��1z���霓��u�Y�#��EPmБbƲ���j�.��f����s^$���"i�&'�4l���u�����Uă��>fo�r$9�!�]��N^��	fk=�����Ͱ�B�mdRw�X@��l�2��=�gU�I|�!�[�8�GD8$d,�0^}�5l�=�p�6Z��(`[WdtE�)֪�+�|q���tǕv`�-߲P�Y�[��8HR���h�+�g��}2��d�M�[Wڹ!�)3	�Ă�M�b6��I��M�}���zz�[�0��2�ء�~���BX��r/M9��H�/^(V�q�*��艫��y�k��n���^��+#B��W*��O��f�V��.�C���F� ����+4�sr/�R�hĢ�F+�� [|�J��V?��u�����/�BN�V%/X�N*sH>b|��a뱁q=�������X[l�T��VRg����8%�
��S%�����ܠ1u�l?V�sC���+�b������Z;q�z�FblPiQ#;$�w�Wh5Z� ��Δ�ah���2r�mWΫ�e�a�38qX��U�.�Ll����p0is<���	[ ڔ��Nc����2��5z!�ְ�,���_=�����j���dQF+jm�lBL�x5F�C�.���>��n��"(���\�p�x0Ӂ��ɵ��E�m�y!�B-HC4���B�'��m�EGv)=��6�ҳk��~�x��=h(�d�bx8fC�O���H��&+׼�D#o�r�ٖ=ޡ?L��
�B����:2!��+�7�"���]!���`ן[U�γ۝غ�!�J2i�f�,|������C�1(��$�*JNA�V����+ԙ5�,��N���rb+I�tQ!Ҋ�h+ڮQ��5dV�s�4�2���P"�R	�T��7�P|X�`�E)1[�H{���
I���v����4�z���d6~�l(��ų��w|+>^��{����ǉ��6�;�2�;��'��X�����[C��-�;{��D���7��s�ˋ�!�^kºS�����#���!����;��~{��p�ű����5DɄ�w����|t\�9�C��놏�4��Mt��3�G,b�O����m�W���*���4!煭1���Q+nCw�I��\U �w[�7�ح~����`��n����x�$�v��z<b�l�sS�Ro��wV<��
uDkbw�ֺ�,08��6�q3�'ͦjQ�h:^�$�ڕ��5*w�fs@������I�z(�ă�!�#��W�Z�{��=dW;�i��l�����U���}l~3[��9�Q,�,�&y�;a|�V��=L�xX�� =����D�	����9��l�t9S�~���[R=$;����B�d���)&��L8��~P�\a�yv�Z���z{��h_n�JoA��	�I���^o�7%Ǒ38l�`%� b�OH�g��p]�{(��H��ĺPa�_!�(�Qlna��]��K�~ZU���ʺ����C�sр��fA����w�6��a�ŽX<������
�����,��噺l�Y�� K'J4<��7b�:C��nIlq~{ڔW��YB��=!��Y߱lL�ae�h�A�Z���+�eC��r/�}��^�`W��Zq�d#�eQ�5�`V�	be}�}��;��ʥ{��d�[0���r�C�d����
���aȽp+�XqN)�¤�&IA�TU(�r_*�	�Sݒ�$EBK���K�=Y��)lD.���w�
Lr=�H��ƙU��
�O�*��-��ѳ�W?�N�q{����D�(/9�l�y�&Q���d��N��k
�m��ǭ+"AR���ݿJH��>\��@��֡$`��ě8uΈ��XwW�ԟh)�<E���; z_d*����|bi�3��*EU�=�.I����J��]�}�:Eԡ@�l_1�LRR<�tǕv`h�ȷ��`�EQb��J��,*��ʧIK57����چ!C��&�@�c����H��T.���>ꇗ.�>-���sJ6W�Hf���?��vt�~�㊄>�6w���J$�)�n#��h�R�����E*�|�p�ֲJP��{x�p�E��G�"�&�R��*mO���a�0����62�@d�r����r��\����e��'\*Zi8dg� !Kx�cH���l[�u�&1�������i��%������<9���=��J'f	�g&���%5��"��o���ⱁС�wvVT�TZ�4�쳔i�@Y�����%��/��	��>��s�G���6'.NXɩd3j^m�E�K�A��V���K#+Ɍ.UW�:�ztRȤ s���.������u ȩ���"s� +
qL9q�(��W���w~������hK	�[�2dLX�R���?��y�y^u�HjB�P*n� ���~o"��mZ��%z��4J~Jx���zc�ъ�iKI�]AJ�h�䈇חV���U)od#����vO���tpE1Cpݷ��F����s.���K6#�%CDݩv%؆��k�d�b**[��2Bz�_���~����=�e0�̈Uu�Y�W���X��L�Ow\�[�s�[����i��W�B΢S4k	7!�M���Ό�o8�)D�}0dpI+�VRS�2���8zJ#2�H�:X�Q��T�d���d�������ު�+�-I<�׈-�h��
�U�Ůb�Ҝμ�O��~����>�+gL0��4n�8R�A,��%<��W�����T�+��]��`sE��v:��ds�ULg���Dv|�b�H-%^��� R-FB��Qg��Ш9�IC
���� D�ě�d��(q�j�Hr������jp���5Fa��xƸA�:�<��D���:#K\�b_�LT� ����"{9h�y��汱�'㑌8I�	�13�z�Ku��DS5S�_p8p$s�*��YܺB�j����@68�������W��E���/I�$�y�����V���i��� q���>BR�4�Z �  �0rY[<K�;1�pdW
g%C6�,�f5��%k��XRy���3f�_��hm�{��E��$���%�7X���t�O�>�tBkl`�	Z�}�.Nݦ�]��'���Ȏ��;�Աr䑙-4�>E��V�4��+���F^�����k'��89�&$�M�[g��F#�Nv �B>�c�D�L&�����n��x#�H�Q�P�+c.�!u����I�Eպ�BTA������9ʩPUpu�#T 0Q�_��g�)Sl��WG�r�吔r�	\�\dW�w�V�Z���q����%���@���y>�����_v³��'H3C-�S�5J�U���K�FD͒�ZH�eKܱ�DϽK	R5��`�1K|4kI�PR�a1ԟTH��Zm3ߜ�U������ �� �P}1���� N7-�r�j�$Q�%�y���l�9�>n����JC�����"�HT8�A��b�WH�6���{I��v��SK����?�Ҡ���%պ8��U�brE��́�X(��ݷ0�V�����F8���3�-���@����6%I�FƄ� ���<A��'�ms����B�fk��t#������:��-�ԏ*����Vj��H[@�%�8�\'����'��>7�f�5PoI���F�4��G�:9a�w��3�����'H���el��.��A�rR���E�y9�Z�)�.�6w,����Ѽ�F�&�F�B�)x�՟T�Z�<䀩 	&X�R�Y�����{�O2�k�k�B��m�gM���zb��ߔ-���CEV�Fs�ϸF�َ7G;���|�W�_�hɆ�R��'�r�$��
%i��~�o��Ya�I?�Zxc�TZw5�����٥q�t;����d�E�/�Ґ��E�w�����.���R|;�t��t�&��iC�A)H,e�V�R��3����Թ��u�:�A)���� ��{�oS�ɩ�F�'Yt/���T�ef�$}��h|.�J{R��R�9�E�������^�I����� �8PL���&����T�]'����hʥ�g����r���Kv�Bĉ��3�;��x`�x���D
@�3�����ѽ�ۃ�F�-��Qr��[bŊ�H�ȢRq���V���^�u�oS��9)�%sr��%hv8��5��=�7dqJA�$o���(�)�]�/��uS)�-�KZ��3ղs�6-J�ך�� �3���E#�]*s�0z���@Xo�haw��ɚ5@�H�~1{�Z�c��� ��*m YI���Uc�Zz��	�Ǥ���A|�ʒ����E�n�O��CN)`�3�'� 2b���&twI��P�d�a���M`�K"�,(��J��*�����7����Jb�-��撝P��Q[n3߫J]�a�����a���SI7C1���H�1� H�2�>�#!����w��?����      Z      x�3202�2202����� �J      T     x�US�n�0<�_�H �)Q�c_� )� ���&���>��[ ��&��L��gH��ў�����s~j��|x������H,���1�I*�X�,�U7���*��JO%U�{�bR��H�/ݕ��͠�z�/k{��i?~ۿ�o��3��I����>�Ț�d�P,� W�|��*��Y*\m3w����aލ���㩿��<�#��G��!Q.0��!2[���>/�u��l����2��3!�8��@�d��r�SyM��{�h�CQ.�4ws��2�v]�h"P�zOQY � 7��)�Kj�*�Ϛ�B�=L�����vZ�}��dE����P�R�ػ� F�k'o��  ��r�A������N��C��xJ��~|={�����[0�w�/�@1�3
�]vS����g��s-�T3Wy�W�N��y�T�u	�
�f��s;�������@�`����r)�b,�&$�P���ET�:�Z�޶�iE�����k�h�SP���+��)k�x�l��������YO���B8o��~�0� ��     