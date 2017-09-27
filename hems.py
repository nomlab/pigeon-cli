import random
import pandas as pd
from operator import attrgetter

class HemsCollection:
    def __init__(self, lst, hems_id, lat, lng):
        self.hems_id = hems_id
        self.contents = [Hems(x) for x in lst if x['frameworx:hemsId'] == hems_id]
        self.lat = self.contents[0].raw.get('frameworx:lat') or lat
        self.lng = self.contents[0].raw.get('frameworx:lng') or lng
        self.calc_powers()

    def calc_powers(self):
        # 発電しているデータのインデックスを抽出
        index_lst = [index for index, x in enumerate(self.contents) if x.charge_flag]
        if len(index_lst) <= 0:
            return

        # 連続して発電している区間をグループ化
        # 区間は発電を行ったデータの±1時間
        # [2, 3, 5] -> [range(1, 5), range(4, 7)]
        pre_index = index_lst[0] - 2
        index_lst.append(index_lst[-1] + 2) # dummy element
        charged_sections = []
        continuous_group = []
        for index in index_lst:
            if (index - pre_index) > 1:
                if len(continuous_group) > 0:
                    charged_sections.append(range(continuous_group[0] - 1, continuous_group[-1] + 2))
                continuous_group = [index]
                pre_index = index
            else:
                continuous_group.append(index)
                pre_index = index

        for section in charged_sections:
            # 発電区間の総電力量
            sum_p = sum([self.contents[index].power_usage for index in section])

            # 発電区間の±1時間の区間で電力量を補間
            self.interpolate_power(section)

            # 補間した区間の総電力量
            sum_i = sum([self.contents[index].power_usage for index in section])

            # 発電区間の実際の総電力量と補間した総電力量の差分をランダムに配分
            ntests = 100
            rest = sum_p - sum_i
            for i in range(ntests):
                r = random.choice(section)
                self.contents[r].power_usage += rest / ntests

    def interpolate_power(self, section):
        m = (self.contents[section[-1] + 1].power_usage - self.contents[section[0] - 1].power_usage) / (len(section) + 2 - 1)
        for i in section:
            self.contents[i].power_usage = self.contents[i - 1].power_usage + m

    def attributes(self, attr_name):
        attr_lst = []
        f = attrgetter(attr_name)
        return [f(x) for x in self.contents]

    def write_csv(self, filename, attr_names):
        df = pd.DataFrame([])
        for name in attr_names:
            df[name] = self.attributes(name)
        df.to_csv(filename, index = False)

    def attribute(self, obj, attr_name):
        f = attrgetter(attr_name)
        return f(obj)

    def to_hash(self, attr_names):
        # [{'timestamp' => '2016-04-01T00:00:00', 'power_usage' => '10.0'} ...]
        # values = [{name: self.attribute(x, name) for name in attr_names} for x in self.contents]
        # {'timestamp' => ['2016-04-01T00:00:00', ...], 'power_usage' => [...]}
        values = {name: self.attributes(name) for name in attr_names}
        return {'hems_id': self.hems_id, 'lat': self.lat, 'lng': self.lng,
                'values': values}

class Hems:
    def __init__(self, dic):
        self.id                = dic["frameworx:hemsId"]
        self.timestamp         = dic["frameworx:date"]
        self.main              = dic["frameworx:main"] or 0
        self.battery_charge    = dic["frameworx:batteryCharge"] or 0
        self.battery_discharge = dic["frameworx:batteryDischarge"] or 0
        self.fuel_cell         = dic["frameworx:fuelCell"] or 0
        self.solar_generated   = dic["frameworx:solarGenerated"] or 0
        self.solar_sold        = dic["frameworx:solarSold"] or 0
        self.gas_usage         = dic["frameworx:gusUsage"] or 0
        self.water_usage       = dic["frameworx:waterUsage"] or 0
        self.charge_flag       = True if self.battery_charge > 0 else False
        self.power_usage = self.main + self.battery_discharge + self.fuel_cell - self.battery_charge + self.solar_generated - self.solar_sold
        self.raw = dic

    def dump(self):
        print("id: ", self.id)
        print("date: ", self.timestamp)
        print("main: ", self.main)
        print("batteryCharge: ", self.battery_charge)
        print("batteryDischarge: ", self.battery_discharge)
        print("fuelCell: ", self.fuel_cell)
        print("solarGenerated: ", self.solar_generated)
        print("solarSold: ", self.solar_sold)
        print("gasUsage: ", self.gas_usage)
        print("waterUsage: ", self.water_usage)
