import gridrad

"""
1- download_data_4_2_Severe : verileri   indir
2- filter_dataset_location  : bulundukları alana göre filtrele. BU işlem kendi içerisinde filter ve remove_clutter içerir. 
                              istenirse piksel sayısına göre de filtreleme yapılaiblir. Default değeri 1 dir
3- create_numpySubDataSet(long,lat,size,dbz_thr,pixel_num_thr) :piksel sayısı ve dBZ- rüzgar hızına göre filtreleme yap,
                                dBZ degeri dbz_th un üzerinde  piksel_th kadar pikseli olan resimler kalır digerleri  filtrelenir
                                istenen kordinattan istenen boyutta veriseti oluştur
4- create_TexasY_from_GridRad(grid_data_np_path,texas_data_path,nameof_location):griddata zamanına göre  texas - y dosyalarını oluşturuyor

** Grid UTC WTM CentralTime zaman formatlarına sahip.Central Standard Time (CST) is six hours behind Coordinated Universal Time (UTC).  Bu yüzden 
		data_texas=data_texas[data_texas["TIME"] == (grid_time+ timedelta(hours=-6))] kullanıldı.
		grid zamanı sabit tutuldu.

"""


# gridrad.download_data_4_2_Severe() # aynı dosya inemesin kısmını yeniden ekle

# gridrad.filter_dataset_location(102,33,1,600)
# gridrad.plot_filtered_np_folders("./data/data600_max_thr_1")

# gridrad.create_numpySubDataSet(600,128,102,33,1,30,10,"Rees") #dbz_thr is compared with the max value within the area, becasue mean value is generally 8

# gridrad.plot_filtered_np_folders("./data/data128_pixel_thr_30_dbz_thr_30_Rees")
gridrad.create_TexasY_from_GridRad("./data/data128_pixel_thr_30_dbz_thr_30_Rees","./data/texas_time","REES")

# gridrad.squeeze_Xvalues("./data/data128_pixel_thr_30_dbz_thr_30_Rees")

a=3



"""old codes"""

# data = gridrad.read_file('./data/raw_data/nexrad_3d_v4_2_20100120T180000Z.nc')
# data = gridrad.filter(data)
# data = gridrad.remove_clutter(data, skip_weak_ll_echo=1)
# plot = gridrad.plot_image(data,"full")
#
# #"area": "33.65/-102.10/33.55/-102", Rees 33.607 -102.04
# # data_rees=gridrad.filter_area(data,265,275,35,40)
# data_rees=gridrad.filter_area(data,101,103.67,32,34.67) # for rees
# plot = gridrad.plot_image(data_rees,"rees")

