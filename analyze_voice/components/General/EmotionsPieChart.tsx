"use client"

import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts"

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  ChartConfig,
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
} from "@/components/ui/chart"

// Updated chart data for emotions
// const chartData = [
//   { emotion: "anger", count: 222, fill: "#ef4444" }, // Red
//   { emotion: "disgust", count: 150, fill: "#84cc16" }, // Lime
//   { emotion: "fear", count: 189, fill: "#8b5cf6" }, // Purple
//   { emotion: "happy", count: 321, fill: "#f59e0b" }, // Amber
//   { emotion: "neutral", count: 178, fill: "#94a3b8" }, // Slate
//   { emotion: "pleasant_surprised", count: 230, fill: "#06b6d4" }, // Cyan
//   { emotion: "sad", count: 99, fill: "#4f46e5" }, // Indigo
// ]

// Updated chart config for emotions
const chartConfig = {
  count: {
    label: "Count",
  },
  anger: {
    label: "Anger",
    color: "#ef4444", // Red
  },
  disgust: {
    label: "Disgust",
    color: "#84cc16", // Lime
  },
  fear: {
    label: "Fear",
    color: "#8b5cf6", // Purple
  },
  happy: {
    label: "Happy",
    color: "#f59e0b", // Amber
  },
  neutral: {
    label: "Neutral",
    color: "#94a3b8", // Slate
  },
  pleasant_surprised: {
    label: "Pleasant Surprise",
    color: "#06b6d4", // Cyan
  },
  sad: {
    label: "Sad",
    color: "#4f46e5", // Indigo
  },
}

// Custom legend to better format the entries
const CustomLegend = ({ payload }) => {
  return (
    <div className="grid grid-cols-2 gap-x-8 gap-y-2 text-sm mt-4">
      {payload.map((entry, index) => (
        <div key={`item-${index}`} className="flex items-center">
          <div
            className="w-3 h-3 mr-2"
            style={{ backgroundColor: entry.color }}
          />
          <span className="mr-2">{entry.value}</span>
          <span className="text-gray-500">({chartData[index].count})</span>
        </div>
      ))}
    </div>
  )
}

export function EmotionsPieChart({ data }) {
  return (
    <Card className="w-full">
      <CardHeader className="pb-2">
        <CardTitle>Emotion Distribution</CardTitle>
        <CardDescription>Detected emotional responses</CardDescription>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="">
          {/* Chart takes up less space horizontally */}
          <div className="w-full h-72">
            {" "}
            {/* Remove sm:w-1/2 */}
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={data}
                  dataKey="count"
                  nameKey="emotion"
                  cx="50%"
                  cy="50%"
                  outerRadius="90%"
                  labelLine={true}
                  label={({ name, percent }) =>
                    `${(percent * 100).toFixed(0)}%`
                  }
                >
                  {data.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(value, name, props) => [
                    `${value} (${(
                      (value /
                        data.reduce((sum, item) => sum + item.count, 0)) *
                      100
                    ).toFixed(1)}%)`,
                    chartConfig[props.payload.emotion]?.label || name,
                  ]}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {/* Legend takes up the rest */}
          <div className="w-full sm:w-1/2 mt-4 sm:mt-0 pl-0 sm:pl-4">
            <div className="grid grid-cols-1 gap-y-2">
              {data.map((item, index) => (
                <div key={index} className="w-full flex justify-between">
                  <div
                    className="flex w-4 h-4 mr-2 rounded-sm"
                    style={{ backgroundColor: item.fill }}
                  />
                  <span className="mr-2">
                    {chartConfig[item.emotion]?.label || item.emotion}
                  </span>
                  <span className="ml-auto font-medium">{item.count}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
