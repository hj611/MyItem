##############################################################
SMART Data

fileName : smartId
https://en.wikipedia.org/wiki/S.M.A.R.T.

serial_number           string	    0
smart_raw_value	        string	    1
smart_value	            string	    2
datestamp	            string	    3


##############################################################
Performance Data

fileName : viewId, metrixId

time        0
server_id   1
value       2

##############################################################
error_disk Data

order_id	bigint	0
nodegroup_id	bigint	1
serialno	string	2
host_name_id	string	3
company_id	bigint	4
site_id	bigint	@desc	5
error_type	string	6
state	string	@desc	7
raw_data	string	8
new_disk_sn	string	9
disk_manufacturer	10
disk_model	string	11
disk_type	string	12
disk_size	bigint	13
user_id	bigint	@desc	14
gmt_fix	datetime	15
gmt_modified	datetime	16
is_deleted	string	17
sys_old_sn	string	18
site_old_sn	string	19
bizdate	string	20

##############################################################
disklocation\disk

sn	0
nodename	1
nodegroup	2
firmware_revision	3
serial_num	4
size	5
manufacturer	6
slot	7
model	8
type	9
physical_status	10
disk_name	11

##############################################################
disklocation\device

id	0
device_type	1
.manifest	2
nodename	3
sn	4
nodegroup	5
state	6
app_state	7
use_state	8
manager_state	9
parent_service_tag	10
site_id	11
vmparent	12
date_outwarranty	13
sm_name	14
idc_id	15
room	16
rack	17
location_in_rack	18
hw_cpu	19
hw_harddisk	20
hw_mem	21
hw_raid	22
mac0	23
mac1	24
asset_number	25
aliid	26
date_purchase	27
app_use_type	28
create_time	29
modify_time	30
group_id	31
product_id	32
product_name	33
parent_name	34
real_hostname	35

##############################################################

The performance data file named "viewID, metricID.txt" (e.g., 16,41.txt) represents a feature, and we can know the actual meaning of "viewID" and "metricID" by querying "viewmeta.txt" file and "metricmeta.txt" file that both under the "PerformanceMeta" folder (we call these two files as "PerformanceMeta/viewmeta.txt" and "PerformanceMeta/metricmeta.txt" in the later description). In general, "metricID" is used to represent a specific performance metric, and "viewID" is used to stand for the corresponding metric categories. 

Each performance data file (e.g., 16,41.txt) contains three data items, i.e., "time", "server_id, and "value".  As to the "value" column, the metric value could be the value of disk-level metrics or the value of server-level metrics, and we can know if it's a disk-level metric or not by quarying "PerformanceMeta / metricmeta.txt" - in the "PerformanceMeta / metricmeta.txt", the metric name starting with "disk" or "Disk" (e.g., disk1) indicates that this metric is the disk-level metric, and the rest of the metrics are the server-level metrics. For disk-level metrics,  "Disk Mount Point" and "Metric Name" have the following correspondence relationship:

Disk Mount Point     Metric Name
/dev/sda, 		   disk1
/dev/sdb,		   disk2
/dev/sdc,		   disk3
/dev/sdd, 		   disk4
/dev/sde, 		   disk5
/dev/sdf, 		   disk6
/dev/sdg, 		   disk7
/dev/sdh, 		   disk8
..., ...

According to the above correspondence relationship, we can know the specific mount point folder path assigned to a specific disk (e.g., /dev.sda). Besides, given a specific "server_id", we can quary and know the corresponding server's serial number from the file named "servermeta.txt" under "PerformanceMeta" folder. Hence, for disk-level metrics, we can know two useful information: (1) the serial number of the corresponding server that this disk is located at, and (2) the specific mount point folder path assigned to this disk. Now, based on the two information, we could know the disk's serial number by quarying the files under "disklocation/disk" folder. Once we have the disk's serial number, we can combine the performance data with SMART data since SMART data is only focus on disk-level. What's more, we can figure out the healthy status of a disk by quarying "error_disk" file, which is also only focus on disk-level. Therefore, it is necessary to know the serial number of a disk if we want to combine performance data and SMART data while knowing the healthy status of disks.

What's more, for each server, we can know the rack, room, and site information by quarying the files under "disklocation / device" folder.