package com.nvidia.spark.rapids.jni;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.List;

/**
 * Used to hold timezone info read from `java.util.TimeZone`
 * This class is used for ORC timezone conversion.
 * For the other timezone conversions, it uses `java.time.ZoneId` APIs.
 * The information is generated from OpenJDK 8. So some timezones in newer JDKs are missing.
 * The reason why we do not read timezone info directly from `java.util.TimeZone`:
 * `sun.util.calendar.ZoneInfo` is not public API, on some JDK distributions (like Oracle JDK),
 * it's not accessible, E.g.: report error: package sun.util.calendar is not visible
 */
class OrcTimezoneInfo {
  public OrcTimezoneInfo(int rawOffset, long[] transitions, int[] offsets) {
    this.rawOffset = rawOffset;
    this.transitions = transitions;
    this.offsets = offsets;
  }

  // in milliseconds
  int rawOffset;

  // in milliseconds
  long[] transitions;

  // in milliseconds
  int[] offsets;

  @Override
  public String toString() {
    return "OrcTimezoneInfo{" +
        "rawOffset=" + rawOffset +
        ", transitions=" + Arrays.toString(transitions) +
        ", offsets=" + Arrays.toString(offsets) +
        '}';
  }

  // The following is Static fields and methods.
  // The `orc_timezone_info.data` file is generated from `sun.util.calendar.ZoneInfo` on OpenJDK 8
  // It first reads `transitions` and `offsets` fields from `ZoneInfo` via reflection.
  // Then calculate the actual transition and offset values via:
  // - actual transition = transition >> 12
  // - actual offset = offsets[transition & 0x0FL]
  // For more details, please refer to `sun.util.calendar.ZoneInfo` source code.

  // Refer to `serializeTimezoneInfo` method for how to generate the file.
  private static final String ORC_TIMEZONE_FILE = "orc_timezone_info.data";

  // the mapped memory for the file
  private static MappedByteBuffer serializedBuf = null;

  static {
    readTimezoneInfoFromFile();
  }

  private static void readTimezoneInfoFromFile() {
    URL path = OrcTimezoneInfo.class.getClassLoader().getResource(ORC_TIMEZONE_FILE);
    if (path == null) {
      throw new RuntimeException("Can not find ORC timezone info file " + ORC_TIMEZONE_FILE);
    }

    try (RandomAccessFile file = new RandomAccessFile(path.getPath(), "r");
         FileChannel fileChannel = file.getChannel()) {

      if (fileChannel.size() > 2 * 1024 * 1024) { // > 2M
        throw new RuntimeException("Failed to load ORC timezone info, file is too large > 2M.");
      }

      // Map the file into memory
      serializedBuf = fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, fileChannel.size());
    } catch (IOException e) {
      throw new RuntimeException("Failed to load ORC timezone info file " + ORC_TIMEZONE_FILE, e);
    }
  }

  /**
   * Get timezone info for the specified timezone Id
   * @param timezoneId timezone Id
   * @return timezone info
   */
  public static OrcTimezoneInfo get(String timezoneId) {
    int index = Arrays.binarySearch(timezoneIds, timezoneId);
    if (index < 0) {
      throw new IllegalArgumentException("Timezone ID not found: " + timezoneId);
    }

    // shallow copy
    ByteBuffer buf = serializedBuf.duplicate();
    buf.order(ByteOrder.BIG_ENDIAN);

    int timezoneInfoOffsetInFile = buf.getInt(Integer.BYTES * index);
    buf.position(timezoneInfoOffsetInFile);

    int rawOffsets = buf.getInt();

    int numTransitions = buf.getInt();
    long[] transitions = new long[numTransitions];
    for (int i = 0; i < numTransitions; ++i) {
      transitions[i] = buf.getLong();
    }

    int numOffsets = buf.getInt();
    int[] offsets = new int[numOffsets];
    for (int i = 0; i < numOffsets; ++i) {
      offsets[i] = buf.getInt();
    }

    return new OrcTimezoneInfo(rawOffsets, transitions, offsets);
  }

  public static List<String> getAllTimezoneIds() {
    return Arrays.asList(timezoneIds);
  }

  private static final String[] timezoneIds = {"ACT", "AET", "AGT", "ART", "AST", "Africa/Abidjan", "Africa/Accra", "Africa/Addis_Ababa", "Africa/Algiers", "Africa/Asmara", "Africa/Asmera", "Africa/Bamako", "Africa/Bangui", "Africa/Banjul", "Africa/Bissau", "Africa/Blantyre", "Africa/Brazzaville", "Africa/Bujumbura", "Africa/Cairo", "Africa/Casablanca", "Africa/Ceuta", "Africa/Conakry", "Africa/Dakar", "Africa/Dar_es_Salaam", "Africa/Djibouti", "Africa/Douala", "Africa/El_Aaiun", "Africa/Freetown", "Africa/Gaborone", "Africa/Harare", "Africa/Johannesburg", "Africa/Juba", "Africa/Kampala", "Africa/Khartoum", "Africa/Kigali", "Africa/Kinshasa", "Africa/Lagos", "Africa/Libreville", "Africa/Lome", "Africa/Luanda", "Africa/Lubumbashi", "Africa/Lusaka", "Africa/Malabo", "Africa/Maputo", "Africa/Maseru", "Africa/Mbabane", "Africa/Mogadishu", "Africa/Monrovia", "Africa/Nairobi", "Africa/Ndjamena", "Africa/Niamey", "Africa/Nouakchott", "Africa/Ouagadougou", "Africa/Porto-Novo", "Africa/Sao_Tome", "Africa/Timbuktu", "Africa/Tripoli", "Africa/Tunis", "Africa/Windhoek", "America/Adak", "America/Anchorage", "America/Anguilla", "America/Antigua", "America/Araguaina", "America/Argentina/Buenos_Aires", "America/Argentina/Catamarca", "America/Argentina/ComodRivadavia", "America/Argentina/Cordoba", "America/Argentina/Jujuy", "America/Argentina/La_Rioja", "America/Argentina/Mendoza", "America/Argentina/Rio_Gallegos", "America/Argentina/Salta", "America/Argentina/San_Juan", "America/Argentina/San_Luis", "America/Argentina/Tucuman", "America/Argentina/Ushuaia", "America/Aruba", "America/Asuncion", "America/Atikokan", "America/Atka", "America/Bahia", "America/Bahia_Banderas", "America/Barbados", "America/Belem", "America/Belize", "America/Blanc-Sablon", "America/Boa_Vista", "America/Bogota", "America/Boise", "America/Buenos_Aires", "America/Cambridge_Bay", "America/Campo_Grande", "America/Cancun", "America/Caracas", "America/Catamarca", "America/Cayenne", "America/Cayman", "America/Chicago", "America/Chihuahua", "America/Ciudad_Juarez", "America/Coral_Harbour", "America/Cordoba", "America/Costa_Rica", "America/Coyhaique", "America/Creston", "America/Cuiaba", "America/Curacao", "America/Danmarkshavn", "America/Dawson", "America/Dawson_Creek", "America/Denver", "America/Detroit", "America/Dominica", "America/Edmonton", "America/Eirunepe", "America/El_Salvador", "America/Ensenada", "America/Fort_Nelson", "America/Fort_Wayne", "America/Fortaleza", "America/Glace_Bay", "America/Godthab", "America/Goose_Bay", "America/Grand_Turk", "America/Grenada", "America/Guadeloupe", "America/Guatemala", "America/Guayaquil", "America/Guyana", "America/Halifax", "America/Havana", "America/Hermosillo", "America/Indiana/Indianapolis", "America/Indiana/Knox", "America/Indiana/Marengo", "America/Indiana/Petersburg", "America/Indiana/Tell_City", "America/Indiana/Vevay", "America/Indiana/Vincennes", "America/Indiana/Winamac", "America/Indianapolis", "America/Inuvik", "America/Iqaluit", "America/Jamaica", "America/Jujuy", "America/Juneau", "America/Kentucky/Louisville", "America/Kentucky/Monticello", "America/Knox_IN", "America/Kralendijk", "America/La_Paz", "America/Lima", "America/Los_Angeles", "America/Louisville", "America/Lower_Princes", "America/Maceio", "America/Managua", "America/Manaus", "America/Marigot", "America/Martinique", "America/Matamoros", "America/Mazatlan", "America/Mendoza", "America/Menominee", "America/Merida", "America/Metlakatla", "America/Mexico_City", "America/Miquelon", "America/Moncton", "America/Monterrey", "America/Montevideo", "America/Montreal", "America/Montserrat", "America/Nassau", "America/New_York", "America/Nipigon", "America/Nome", "America/Noronha", "America/North_Dakota/Beulah", "America/North_Dakota/Center", "America/North_Dakota/New_Salem", "America/Nuuk", "America/Ojinaga", "America/Panama", "America/Pangnirtung", "America/Paramaribo", "America/Phoenix", "America/Port-au-Prince", "America/Port_of_Spain", "America/Porto_Acre", "America/Porto_Velho", "America/Puerto_Rico", "America/Punta_Arenas", "America/Rainy_River", "America/Rankin_Inlet", "America/Recife", "America/Regina", "America/Resolute", "America/Rio_Branco", "America/Rosario", "America/Santa_Isabel", "America/Santarem", "America/Santiago", "America/Santo_Domingo", "America/Sao_Paulo", "America/Scoresbysund", "America/Shiprock", "America/Sitka", "America/St_Barthelemy", "America/St_Johns", "America/St_Kitts", "America/St_Lucia", "America/St_Thomas", "America/St_Vincent", "America/Swift_Current", "America/Tegucigalpa", "America/Thule", "America/Thunder_Bay", "America/Tijuana", "America/Toronto", "America/Tortola", "America/Vancouver", "America/Virgin", "America/Whitehorse", "America/Winnipeg", "America/Yakutat", "America/Yellowknife", "Antarctica/Casey", "Antarctica/Davis", "Antarctica/DumontDUrville", "Antarctica/Macquarie", "Antarctica/Mawson", "Antarctica/McMurdo", "Antarctica/Palmer", "Antarctica/Rothera", "Antarctica/South_Pole", "Antarctica/Syowa", "Antarctica/Troll", "Antarctica/Vostok", "Arctic/Longyearbyen", "Asia/Aden", "Asia/Almaty", "Asia/Amman", "Asia/Anadyr", "Asia/Aqtau", "Asia/Aqtobe", "Asia/Ashgabat", "Asia/Ashkhabad", "Asia/Atyrau", "Asia/Baghdad", "Asia/Bahrain", "Asia/Baku", "Asia/Bangkok", "Asia/Barnaul", "Asia/Beirut", "Asia/Bishkek", "Asia/Brunei", "Asia/Calcutta", "Asia/Chita", "Asia/Choibalsan", "Asia/Chongqing", "Asia/Chungking", "Asia/Colombo", "Asia/Dacca", "Asia/Damascus", "Asia/Dhaka", "Asia/Dili", "Asia/Dubai", "Asia/Dushanbe", "Asia/Famagusta", "Asia/Gaza", "Asia/Harbin", "Asia/Hebron", "Asia/Ho_Chi_Minh", "Asia/Hong_Kong", "Asia/Hovd", "Asia/Irkutsk", "Asia/Istanbul", "Asia/Jakarta", "Asia/Jayapura", "Asia/Jerusalem", "Asia/Kabul", "Asia/Kamchatka", "Asia/Karachi", "Asia/Kashgar", "Asia/Kathmandu", "Asia/Katmandu", "Asia/Khandyga", "Asia/Kolkata", "Asia/Krasnoyarsk", "Asia/Kuala_Lumpur", "Asia/Kuching", "Asia/Kuwait", "Asia/Macao", "Asia/Macau", "Asia/Magadan", "Asia/Makassar", "Asia/Manila", "Asia/Muscat", "Asia/Nicosia", "Asia/Novokuznetsk", "Asia/Novosibirsk", "Asia/Omsk", "Asia/Oral", "Asia/Phnom_Penh", "Asia/Pontianak", "Asia/Pyongyang", "Asia/Qatar", "Asia/Qostanay", "Asia/Qyzylorda", "Asia/Rangoon", "Asia/Riyadh", "Asia/Saigon", "Asia/Sakhalin", "Asia/Samarkand", "Asia/Seoul", "Asia/Shanghai", "Asia/Singapore", "Asia/Srednekolymsk", "Asia/Taipei", "Asia/Tashkent", "Asia/Tbilisi", "Asia/Tehran", "Asia/Tel_Aviv", "Asia/Thimbu", "Asia/Thimphu", "Asia/Tokyo", "Asia/Tomsk", "Asia/Ujung_Pandang", "Asia/Ulaanbaatar", "Asia/Ulan_Bator", "Asia/Urumqi", "Asia/Ust-Nera", "Asia/Vientiane", "Asia/Vladivostok", "Asia/Yakutsk", "Asia/Yangon", "Asia/Yekaterinburg", "Asia/Yerevan", "Atlantic/Azores", "Atlantic/Bermuda", "Atlantic/Canary", "Atlantic/Cape_Verde", "Atlantic/Faeroe", "Atlantic/Faroe", "Atlantic/Jan_Mayen", "Atlantic/Madeira", "Atlantic/Reykjavik", "Atlantic/South_Georgia", "Atlantic/St_Helena", "Atlantic/Stanley", "Australia/ACT", "Australia/Adelaide", "Australia/Brisbane", "Australia/Broken_Hill", "Australia/Canberra", "Australia/Currie", "Australia/Darwin", "Australia/Eucla", "Australia/Hobart", "Australia/LHI", "Australia/Lindeman", "Australia/Lord_Howe", "Australia/Melbourne", "Australia/NSW", "Australia/North", "Australia/Perth", "Australia/Queensland", "Australia/South", "Australia/Sydney", "Australia/Tasmania", "Australia/Victoria", "Australia/West", "Australia/Yancowinna", "BET", "BST", "Brazil/Acre", "Brazil/DeNoronha", "Brazil/East", "Brazil/West", "CAT", "CET", "CNT", "CST", "CST6CDT", "CTT", "Canada/Atlantic", "Canada/Central", "Canada/Eastern", "Canada/Mountain", "Canada/Newfoundland", "Canada/Pacific", "Canada/Saskatchewan", "Canada/Yukon", "Chile/Continental", "Chile/EasterIsland", "Cuba", "EAT", "ECT", "EET", "EST", "EST5EDT", "Egypt", "Eire", "Etc/GMT", "Etc/GMT+0", "Etc/GMT+1", "Etc/GMT+10", "Etc/GMT+11", "Etc/GMT+12", "Etc/GMT+2", "Etc/GMT+3", "Etc/GMT+4", "Etc/GMT+5", "Etc/GMT+6", "Etc/GMT+7", "Etc/GMT+8", "Etc/GMT+9", "Etc/GMT-0", "Etc/GMT-1", "Etc/GMT-10", "Etc/GMT-11", "Etc/GMT-12", "Etc/GMT-13", "Etc/GMT-14", "Etc/GMT-2", "Etc/GMT-3", "Etc/GMT-4", "Etc/GMT-5", "Etc/GMT-6", "Etc/GMT-7", "Etc/GMT-8", "Etc/GMT-9", "Etc/GMT0", "Etc/Greenwich", "Etc/UCT", "Etc/UTC", "Etc/Universal", "Etc/Zulu", "Europe/Amsterdam", "Europe/Andorra", "Europe/Astrakhan", "Europe/Athens", "Europe/Belfast", "Europe/Belgrade", "Europe/Berlin", "Europe/Bratislava", "Europe/Brussels", "Europe/Bucharest", "Europe/Budapest", "Europe/Busingen", "Europe/Chisinau", "Europe/Copenhagen", "Europe/Dublin", "Europe/Gibraltar", "Europe/Guernsey", "Europe/Helsinki", "Europe/Isle_of_Man", "Europe/Istanbul", "Europe/Jersey", "Europe/Kaliningrad", "Europe/Kiev", "Europe/Kirov", "Europe/Kyiv", "Europe/Lisbon", "Europe/Ljubljana", "Europe/London", "Europe/Luxembourg", "Europe/Madrid", "Europe/Malta", "Europe/Mariehamn", "Europe/Minsk", "Europe/Monaco", "Europe/Moscow", "Europe/Nicosia", "Europe/Oslo", "Europe/Paris", "Europe/Podgorica", "Europe/Prague", "Europe/Riga", "Europe/Rome", "Europe/Samara", "Europe/San_Marino", "Europe/Sarajevo", "Europe/Saratov", "Europe/Simferopol", "Europe/Skopje", "Europe/Sofia", "Europe/Stockholm", "Europe/Tallinn", "Europe/Tirane", "Europe/Tiraspol", "Europe/Ulyanovsk", "Europe/Uzhgorod", "Europe/Vaduz", "Europe/Vatican", "Europe/Vienna", "Europe/Vilnius", "Europe/Volgograd", "Europe/Warsaw", "Europe/Zagreb", "Europe/Zaporozhye", "Europe/Zurich", "GB", "GB-Eire", "GMT", "GMT0", "Greenwich", "HST", "Hongkong", "IET", "IST", "Iceland", "Indian/Antananarivo", "Indian/Chagos", "Indian/Christmas", "Indian/Cocos", "Indian/Comoro", "Indian/Kerguelen", "Indian/Mahe", "Indian/Maldives", "Indian/Mauritius", "Indian/Mayotte", "Indian/Reunion", "Iran", "Israel", "JST", "Jamaica", "Japan", "Kwajalein", "Libya", "MET", "MIT", "MST", "MST7MDT", "Mexico/BajaNorte", "Mexico/BajaSur", "Mexico/General", "NET", "NST", "NZ", "NZ-CHAT", "Navajo", "PLT", "PNT", "PRC", "PRT", "PST", "PST8PDT", "Pacific/Apia", "Pacific/Auckland", "Pacific/Bougainville", "Pacific/Chatham", "Pacific/Chuuk", "Pacific/Easter", "Pacific/Efate", "Pacific/Enderbury", "Pacific/Fakaofo", "Pacific/Fiji", "Pacific/Funafuti", "Pacific/Galapagos", "Pacific/Gambier", "Pacific/Guadalcanal", "Pacific/Guam", "Pacific/Honolulu", "Pacific/Johnston", "Pacific/Kanton", "Pacific/Kiritimati", "Pacific/Kosrae", "Pacific/Kwajalein", "Pacific/Majuro", "Pacific/Marquesas", "Pacific/Midway", "Pacific/Nauru", "Pacific/Niue", "Pacific/Norfolk", "Pacific/Noumea", "Pacific/Pago_Pago", "Pacific/Palau", "Pacific/Pitcairn", "Pacific/Pohnpei", "Pacific/Ponape", "Pacific/Port_Moresby", "Pacific/Rarotonga", "Pacific/Saipan", "Pacific/Samoa", "Pacific/Tahiti", "Pacific/Tarawa", "Pacific/Tongatapu", "Pacific/Truk", "Pacific/Wake", "Pacific/Wallis", "Pacific/Yap", "Poland", "Portugal", "ROK", "SST", "Singapore", "SystemV/AST4", "SystemV/AST4ADT", "SystemV/CST6", "SystemV/CST6CDT", "SystemV/EST5", "SystemV/EST5EDT", "SystemV/HST10", "SystemV/MST7", "SystemV/MST7MDT", "SystemV/PST8", "SystemV/PST8PDT", "SystemV/YST9", "SystemV/YST9YDT", "Turkey", "UCT", "US/Alaska", "US/Aleutian", "US/Arizona", "US/Central", "US/East-Indiana", "US/Eastern", "US/Hawaii", "US/Indiana-Starke", "US/Michigan", "US/Mountain", "US/Pacific", "US/Samoa", "UTC", "Universal", "VST", "W-SU", "WET", "Zulu"};

  /**
   * This method is only used to generate the timezone info file for maintenance purpose.
   *
   * The generated file is based on OpenJDK 8's `sun.util.calendar.ZoneInfo` implementation.
   * Since `ZoneInfo` is not public API, on some JDK distributions (like Oracle JDK),
   * it's not accessible. So we comment the method out to avoid build issues.
   *
   * File format:
   * - First N * 4 bytes: N is number of timezone Ids
   *   - each 4 bytes is the offset of the timezone info in the file
   * - Then each timezone info:
   *   - 4 bytes: rawOffset (int)
   *   - 4 bytes: numTransitions (int)
   *   - numTransitions * 8 bytes: transitions (long[])
   *   - 4 bytes: numOffsets (int)
   *   - numOffsets * 4 bytes: offsets (int[])
   *
   * How to do the maintenance:
   * - update the `timezoneIds` via TimeZone.getAvailableIDs() and sort them.
   * - run this method to generate the timezone info file, and copy the file to resources folder.
   */
  public static void serializeTimezoneInfo() {
//    try {
//      String path = "/tmp/orc_timezone_info.data";
//
//      // sort timezone ids
//      String[] ids = TimeZone.getAvailableIDs();
//      ArrayList<String> sortedIds = new ArrayList<>(Arrays.asList(ids));
//      sortedIds.sort(String::compareTo);
//
//      List<Integer> timezoneOffsets = new ArrayList<>();
//      DataOutputStream out = new DataOutputStream(Files.newOutputStream(Paths.get(path)));
//
//      // from ZoneInfo source code
//      long OFFSET_MASK_IN_ZONE_INFO = 0x0FL;
//      int TRANSITION_NSHIFT_IN_ZONE_INFO = 12;
//
//      // collect offsets for each timezone
//      int timezoneOffsetInFile = 0;
//      for (String id : sortedIds) {
//        timezoneOffsets.add(timezoneOffsetInFile);
//
//        ZoneInfo zoneInfo = (ZoneInfo) TimeZone.getTimeZone(id);
//        long[] trans = (long[]) FieldUtils.readField(zoneInfo, "transitions");
//        int numTransitions = trans == null ? 0 : trans.length;
//
//        // timezone serialized size calculation
//        timezoneOffsetInFile += 4; // rawOffset
//        timezoneOffsetInFile += 4; // numTransitions
//        timezoneOffsetInFile += numTransitions * 8; // transitions longs
//        timezoneOffsetInFile += 4; // numOffsets
//        timezoneOffsetInFile += numTransitions * 4; // offsets ints
//      }
//
//      // First write all timezone offsets in the file
//      int totalOffsetIndicesSize = sortedIds.size() * 4;
//      for (int off : timezoneOffsets) {
//        out.writeInt(off + totalOffsetIndicesSize);
//      }
//
//      // Then write each timezone info
//      for (String id : sortedIds) {
//        ZoneInfo zoneInfo = (ZoneInfo) TimeZone.getTimeZone(id);
//        long[] trans = (long[]) FieldUtils.readField(zoneInfo, "transitions");
//        int[] offs = (int[]) FieldUtils.readField(zoneInfo, "offsets");
//        int rawOff = (int) FieldUtils.readField(zoneInfo, "rawOffset");
//
//        int numTransitions = trans == null ? 0 : trans.length;
//
//        long[] actualTrans = new long[numTransitions];
//        int[] actualOffsets = new int[numTransitions];
//        for (int i = 0; i < numTransitions; ++i) {
//          // `trans` is combination of transition and offset index
//          actualTrans[i] = trans[i] >> TRANSITION_NSHIFT_IN_ZONE_INFO;
//          // the `offs` is a dictionary, get the actual offset value via index
//          // `trans[i] & OFFSET_MASK_IN_ZONE_INFO` is to get offset index
//          actualOffsets[i] = offs[(int) (trans[i] & OFFSET_MASK_IN_ZONE_INFO)];
//        }
//
//        out.writeInt(rawOff);
//
//        out.writeInt(numTransitions);
//        for (long t : actualTrans) {
//          out.writeLong(t);
//        }
//
//        out.writeInt(numTransitions);
//        for (int o : actualOffsets) {
//          out.writeInt(o);
//        }
//      }
//      out.flush();
//      out.close();
//    } catch (Exception e) {
//      throw new RuntimeException("Failed to serialize ORC timezone info.", e);
//    }
  }
}
