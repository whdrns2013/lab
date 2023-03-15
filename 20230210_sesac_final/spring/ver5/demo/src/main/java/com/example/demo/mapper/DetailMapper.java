package com.example.demo.mapper;

import java.util.ArrayList;

import org.apache.ibatis.annotations.Mapper;

import com.example.demo.model.DetailList;

@Mapper
public interface DetailMapper {

    public ArrayList<DetailList> select(String event_name);
    
}
