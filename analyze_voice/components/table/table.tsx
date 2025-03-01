"use client"

import * as React from "react"
import { useRouter } from "next/navigation"
// Updated to use next/navigation for app router
import {
  ColumnDef,
  ColumnFiltersState,
  SortingState,
  VisibilityState,
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  useReactTable,
} from "@tanstack/react-table"
import { ArrowUpDown, ChevronDown, MoreHorizontal, Plus } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Input } from "@/components/ui/input"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"

import { AddUser } from "../General/AddUser"

const data: Payment[] = [
  {
    id: "m5gr84i9",
    phone: "316-555-0123",
    status: "success",
    date: new Date("December 17, 1995 03:24:00"),
    name: "Ken Adams",
  },
  {
    id: "3u1reuv4",
    phone: "242-555-0198",
    status: "success",
    date: new Date("March 5, 2001 10:15:30"),
    name: "Abe Lincoln",
  },
  {
    id: "derv1ws0",
    phone: "837-555-0147",
    status: "pending",
    date: new Date("June 12, 2015 14:45:00"),
    name: "Monserrat Rivera",
  },
  {
    id: "5kma53ae",
    phone: "874-555-0162",
    status: "success",
    date: new Date("August 21, 2018 08:30:00"),
    name: "Silas Thompson",
  },
  {
    id: "bhqecj4p",
    phone: "721-555-0173",
    status: "failed",
    date: new Date("January 2, 2020 19:20:45"),
    name: "Carmella Johnson",
  },
  {
    id: "y3er8ks2",
    phone: "654-555-0135",
    status: "success",
    date: new Date("November 10, 2013 22:10:00"),
    name: "James Carter",
  },
  {
    id: "p9dm4xw7",
    phone: "235-555-0189",
    status: "success",
    date: new Date("July 18, 2016 12:55:00"),
    name: "Lena Clarkson",
  },
  {
    id: "v8sl3ek0",
    phone: "458-555-0124",
    status: "failed",
    date: new Date("September 29, 2022 06:40:00"),
    name: "Daniel Foster",
  },
  {
    id: "u7kc2eq5",
    phone: "369-555-0158",
    status: "success",
    date: new Date("April 14, 2005 17:25:00"),
    name: "Sophia Martinez",
  },
  {
    id: "t2we9dj3",
    phone: "589-555-0179",
    status: "failed",
    date: new Date("May 30, 2011 20:05:00"),
    name: "Olivia Chang",
  },
  {
    id: "a5xz1mw4",
    phone: "902-555-0194",
    status: "failed",
    date: new Date("February 25, 2017 09:10:00"),
    name: "Ethan Wright",
  },
  {
    id: "w6cn9xa8",
    phone: "103-555-0183",
    status: "success",
    date: new Date("October 8, 2019 15:45:00"),
    name: "Mia Anderson",
  },
  {
    id: "b4yx7qp9",
    phone: "741-555-0119",
    status: "pending",
    date: new Date("March 13, 2023 11:30:00"),
    name: "Noah Bennett",
  },
  {
    id: "z9cm5ep2",
    phone: "825-555-0120",
    status: "success",
    date: new Date("December 5, 2008 04:50:00"),
    name: "Emma Lee",
  },
  {
    id: "d2kl8sy3",
    phone: "369-555-0185",
    status: "failed",
    date: new Date("July 7, 2021 13:20:00"),
    name: "Liam Rodriguez",
  },
]

export type Payment = {
  id: string
  phone: string
  date: Date
  status: "pending" | "success" | "failed"
  name: string
}

export function DataTable() {
  const router = useRouter() // Initialize router

  const [sorting, setSorting] = React.useState<SortingState>([])
  const [columnFilters, setColumnFilters] = React.useState<ColumnFiltersState>(
    []
  )
  const [columnVisibility, setColumnVisibility] =
    React.useState<VisibilityState>({})
  const [rowSelection, setRowSelection] = React.useState({})

  // Define columns inside the component to access router
  const columns: ColumnDef<Payment>[] = [
    {
      id: "select",
      header: ({ table }) => (
        <Checkbox
          checked={
            table.getIsAllPageRowsSelected() ||
            (table.getIsSomePageRowsSelected() && "indeterminate")
          }
          onCheckedChange={(value) => table.toggleAllPageRowsSelected(!!value)}
          aria-label="Select all"
        />
      ),
      cell: ({ row }) => (
        <Checkbox
          checked={row.getIsSelected()}
          onCheckedChange={(value) => row.toggleSelected(!!value)}
          aria-label="Select row"
        />
      ),
      enableSorting: false,
      enableHiding: false,
    },
    {
      accessorKey: "status",
      header: "Status",
      cell: ({ row }) => (
        <div className="capitalize">{row.getValue("status")}</div>
      ),
    },
    {
      accessorKey: "name",
      header: ({ column }) => {
        return (
          <Button
            variant="ghost"
            onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
          >
            Name
            <ArrowUpDown className="ml-2 h-4 w-4" />
          </Button>
        )
      },
      cell: ({ row }) => <div>{row.getValue("name")}</div>,
    },
    {
      accessorKey: "date",
      header: ({ column }) => {
        return (
          <Button
            variant="ghost"
            onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
          >
            Call Date
            <ArrowUpDown className="ml-2 h-4 w-4" />
          </Button>
        )
      },
      cell: ({ row }) => {
        const dateValue = row.getValue<Date>("date")
        const status = row.getValue("status")

        const formattedDate = new Intl.DateTimeFormat("en-US", {
          day: "numeric",
          month: "short",
          year: "numeric",
        }).format(dateValue)

        return (
          <div>
            {status == "success"
              ? formattedDate
              : status == "pending"
              ? "No call made"
              : "Not able to connect call"}
          </div>
        )
      },
    },
    {
      accessorKey: "phone",
      header: () => <div className="text-right">Phone</div>,
      cell: ({ row }) => {
        return (
          <div className="text-right font-medium">{row.getValue("phone")}</div>
        )
      },
    },
    {
      id: "actions",
      enableHiding: false,
      cell: ({ row }) => {
        const user = row.original

        return (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" className="h-8 w-8 p-0">
                <span className="sr-only">Open menu</span>
                <MoreHorizontal className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuLabel>Actions</DropdownMenuLabel>
              <DropdownMenuItem onClick={() => router.push(`/${user.phone}`)}>
                View user information
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        )
      },
    },
  ]

  const table = useReactTable({
    data,
    columns,
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    getCoreRowModel: getCoreRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    onColumnVisibilityChange: setColumnVisibility,
    onRowSelectionChange: setRowSelection,
    state: {
      sorting,
      columnFilters,
      columnVisibility,
      rowSelection,
    },
  })

  return (
    <div className="w-full">
      <div className="flex items-center py-4">
        <div className="flex gap-4 w-1/2">
          <AddUser />
          <Input
            placeholder="Filter names..."
            value={(table.getColumn("name")?.getFilterValue() as string) ?? ""}
            onChange={(event) =>
              table.getColumn("name")?.setFilterValue(event.target.value)
            }
          />
        </div>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline" className="ml-auto">
              Columns <ChevronDown className="ml-2 h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            {table
              .getAllColumns()
              .filter((column) => column.getCanHide())
              .map((column) => {
                return (
                  <DropdownMenuCheckboxItem
                    key={column.id}
                    className="capitalize"
                    checked={column.getIsVisible()}
                    onCheckedChange={(value) =>
                      column.toggleVisibility(!!value)
                    }
                  >
                    {column.id}
                  </DropdownMenuCheckboxItem>
                )
              })}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
      <div className="rounded-md border">
        <Table>
          <TableHeader>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => {
                  return (
                    <TableHead key={header.id}>
                      {header.isPlaceholder
                        ? null
                        : flexRender(
                            header.column.columnDef.header,
                            header.getContext()
                          )}
                    </TableHead>
                  )
                })}
              </TableRow>
            ))}
          </TableHeader>
          <TableBody>
            {table.getRowModel().rows?.length ? (
              table.getRowModel().rows.map((row) => (
                <TableRow
                  key={row.id}
                  data-state={row.getIsSelected() && "selected"}
                >
                  {row.getVisibleCells().map((cell) => (
                    <TableCell key={cell.id}>
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext()
                      )}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell
                  colSpan={columns.length}
                  className="h-24 text-center"
                >
                  No results.
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>
      <div className="flex items-center justify-end space-x-2 py-4">
        <div className="flex-1 text-sm text-muted-foreground">
          {table.getFilteredSelectedRowModel().rows.length} of{" "}
          {table.getFilteredRowModel().rows.length} row(s) selected.
        </div>
        <div className="space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => table.previousPage()}
            disabled={!table.getCanPreviousPage()}
          >
            Previous
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => table.nextPage()}
            disabled={!table.getCanNextPage()}
          >
            Next
          </Button>
        </div>
      </div>
    </div>
  )
}
